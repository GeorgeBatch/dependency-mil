import json
import os.path

import torch
import torch.nn as nn
import torch.nn.functional as F

from source.models.class_connectors import BahdanauSelfAttention, TransformerSelfAttention
from source.models.classifier_heads import LinearClassifier, DSConvClassifier, CommunicatingConvClassifier


class MILClassifier(nn.Module):
    def __init__(self,
                 instance_embedder: nn.Module = None,
                 bag_aggregator: nn.Module = None,
                 class_connector: nn.Module = None,
                 classifier: nn.Module = None
                 ) -> None:
        super(MILClassifier, self).__init__()

        self.instance_embedder = instance_embedder
        self.bag_aggregator = bag_aggregator
        self.class_connector = class_connector
        self.classifier = classifier
        self.init_weights()

    def forward(self, x, bag_lens=None):
        device = next(self.parameters()).device
        x = x.to(device)

        feats_ins = self.compute_instance_embeds(x)
        att_bag, feat_bag, aux = self.bag_aggregator(feats_ins, bag_lens)
        feat_bag = self.class_connector(
            feat_bag)  # feat_bag: (batch, num_classes, embedding_size) -> feat_bag: (batch, num_classes, embedding_size)
        pred_bag = self.classifier(
            feat_bag)  # feat_bag: (batch, num_classes, embedding_size) -> pred_bag: (batch, num_classes)

        return pred_bag, att_bag, feat_bag, aux

    def compute_instance_embeds(self, x):

        # for images
        # stack = x.view(x.shape[0]*x.shape[1], 1, x.shape[2], x.shape[3])
        # for features
        stack = x.view(x.shape[0] * x.shape[1], 1, x.shape[2])

        stack_feat = self.instance_embedder(stack)
        stack_feat = stack_feat.view(x.shape[0], x.shape[1], -1)
        return stack_feat

    def init_weights(self):
        if isinstance(self, nn.Linear):
            torch.nn.init.xavier_normal_(self.weight)
            self.bias.data.fill_(0.01)

        if isinstance(self, nn.Conv2d):
            nn.init.kaiming_uniform_(self, mode='fan_in')


# ------------------ MIL Aggregators --------------------

class AbmilBagClassifier(nn.Module):
    def __init__(self,
                 gated: bool = False,
                 num_classes: int = 1,
                 embedding_size: int = 512,
                 proj_size: int = 128,
                 preproc_net: nn.Module = nn.Identity()
                 ) -> None:
        super(AbmilBagClassifier, self).__init__()

        self.preproc = preproc_net
        self.gated = gated

        if not gated:
            self.attention = nn.Sequential(
                nn.Linear(embedding_size, proj_size),
                nn.PReLU(),
                nn.Linear(proj_size, num_classes)
            )
        else:
            self.attention_v = nn.Sequential(
                nn.Linear(embedding_size, proj_size),
                nn.Tanh()
            )
            self.attention_u = nn.Sequential(
                nn.Linear(embedding_size, proj_size),
                nn.Sigmoid()
            )
            self.attention = nn.Linear(proj_size, num_classes)

    def forward(self, x, bag_lens=None):
        # compute masks
        device = x.device
        x = self.preproc(x)  # batch x max_bag_length x embedding_size
        if not self.gated:
            A = self.attention(x)  # batch x max_bag_length x num_classes
        else:
            A_v = self.attention_v(x)  # batch x max_bag_length x proj_size
            A_u = self.attention_u(x)  # batch x max_bag_length x proj_size
            # batch x max_bag_length x num_classes
            A = self.attention(A_v * A_u)

        mask = torch.zeros((x.shape[0], x.shape[1], A.shape[-1]),
                           device=device, dtype=torch.bool)  # batch x bag x 1
        for i in range(mask.shape[0]):
            mask[i, :bag_lens[i], :] = True
        A[~mask] = float('-inf')  # apply mask
        # over instances in a bag, batch x bag x num_classes
        A = F.softmax(A, 1)
        # batch x num_classes x embedding_size
        B = torch.bmm(torch.permute(A, (0, 2, 1)), x)

        return A, B, None


class DsmilBagClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 embedding_size: int = 512,
                 q_size: int = 128,
                 q_nonlinear: bool = True,
                 v_size: int = 512,
                 v_identity: bool = True,
                 preproc_net: nn.Module = nn.Identity()
                 ) -> None:
        super(DsmilBagClassifier, self).__init__()

        self.preproc = preproc_net

        if q_nonlinear:
            self.q_net = nn.Sequential(
                nn.Linear(embedding_size, q_size),
                nn.ReLU(),
                nn.Linear(q_size, q_size),
                nn.Tanh()
            )
        else:
            self.q_net = nn.Linear(embedding_size, q_size)

        if v_identity:
            assert embedding_size == v_size, f"v_size={v_size} must be equal to embedding_size={embedding_size} when v_identity is True"
            self.v_net = nn.Identity()
        else:
            self.v_net = nn.Sequential(
                nn.Linear(embedding_size, v_size),
                nn.PReLU()
            )

        self.instance_classifier = nn.Linear(embedding_size, num_classes)

    def forward(self, x, bag_lens=None):
        device = x.device
        x = self.preproc(x)  # batch x max_bag_length x embedding_size

        V = self.v_net(x)  # batch x max_bag_length x v_size
        Q = self.q_net(x)  # batch x max_bag_length x q_size

        # batch x max_bag_length x num_classes
        Inst = self.instance_classifier(x)
        for i in range(Inst.shape[0]):
            Inst[i, bag_lens[i]:, :] = float(
                '-inf')  # instanses that are pure padding should never be considered as critical
        I_m, M_id = torch.max(Inst, dim=1)  # batch x num_classes

        Q_m = torch.gather(Q, 1, M_id.unsqueeze(2).expand(M_id.size(0), M_id.size(1),
                                                          Q.size(2)))  # batch x num_classes x q_size

        # batch x max_bag_length x num_classes
        A = torch.bmm(Q, torch.permute(Q_m, (0, 2, 1)))

        real_instances_mask = torch.zeros((x.shape[0], x.shape[1], A.shape[-1]), device=device,
                                          dtype=torch.bool)  # batch x bag x 1
        for i in range(real_instances_mask.shape[0]):
            real_instances_mask[i, :bag_lens[i], :] = True
        # apply mask to all purely padded instances
        A[~real_instances_mask] = float('-inf')

        # over instances in a bag, batch x max_bag_length x num_classes
        A = F.softmax(A, 1)

        # batch x num_classes x v_size
        B = torch.bmm(torch.permute(A, (0, 2, 1)), V)
        return A, B, I_m


# ------------------ How it works together --------------------

def get_model(
        instance_embedder_name: str = "identity",
        bag_aggregator_name: str = "dsmil",
        class_connector_name: str = "identity",
        classifier_name: str = "communicating_conv",
        num_classes: int = 2,
        embedding_size: int = 512,
        proj_size: int = 128,
) -> nn.Module:
    assert instance_embedder_name == "identity", f"instance_embedder_name={instance_embedder_name} is not supported"
    assert bag_aggregator_name in [
        "abmil", "dsmil"], f"bag_aggregator_name={bag_aggregator_name} is not supported"
    assert class_connector_name in ["identity", "bahdanau", "transformer"], \
        f"class_connector_name={class_connector_name} is not supported"
    assert classifier_name in ["linear", "depthwise_separable_conv", "communicating_conv"], \
        f"classifier_name={classifier_name} is not supported"

    if instance_embedder_name == "identity":
        instance_embedder = nn.Identity()
    else:
        raise NotImplementedError(
            f"instance_embedder_name={instance_embedder_name} is not supported")

    if bag_aggregator_name == "abmil":
        bag_aggregator = AbmilBagClassifier(
            num_classes=num_classes,
            embedding_size=embedding_size,
            proj_size=proj_size,
            gated=False,
            preproc_net=nn.Identity()
        )
    elif bag_aggregator_name == "dsmil":
        bag_aggregator = DsmilBagClassifier(
            num_classes=num_classes,
            embedding_size=embedding_size,
            q_size=proj_size,
            v_size=embedding_size,
            v_identity=True,
            preproc_net=nn.Identity()
        )
    else:
        raise NotImplementedError(
            f"bag_aggregator_name={bag_aggregator_name} is not supported")

    if class_connector_name == "identity":
        class_connector = nn.Identity()
    elif class_connector_name == "bahdanau":
        class_connector = BahdanauSelfAttention(embedding_size=embedding_size)
    elif class_connector_name == "transformer":
        class_connector = TransformerSelfAttention(
            embedding_size=embedding_size, num_heads=1)
    else:
        raise NotImplementedError(
            f"class_connector_name={class_connector_name} is not supported")

    if classifier_name == "linear":
        classifier = LinearClassifier(embedding_size=embedding_size)
    elif classifier_name == "depthwise_separable_conv":
        classifier = DSConvClassifier(
            num_classes=num_classes, embedding_size=embedding_size)
    elif classifier_name == "communicating_conv":
        classifier = CommunicatingConvClassifier(
            num_classes=num_classes, embedding_size=embedding_size)
    else:
        raise NotImplementedError(
            f"classifier_name={classifier_name} is not supported")

    milnet = MILClassifier(
        instance_embedder, bag_aggregator, class_connector, classifier)

    return milnet


# ------------------ How it works together --------------------

if __name__ == "__main__":
    print("Testing all models.")

    torch.manual_seed(0)

    sample_num_classes = 2
    sample_input_size = 512
    sample_proj_size = 128
    # sample_bag_lens = [10]
    sample_bag_lens = [6, 7, 8, 10]

    batch_size = len(sample_bag_lens)
    max_bag_length = max(sample_bag_lens)
    input_embedding = torch.rand((batch_size, max_bag_length, sample_input_size), dtype=torch.float32)
    # pretend to pad the input embedding
    for i in range(len(sample_bag_lens)):
        input_embedding[i, sample_bag_lens[i]:, :] = 0

    # ipdb.set_trace()

    layer_correspondence_dict = {}
    for bag_aggregator_name in ['abmil', 'dsmil']:
        for class_connector_name in ['identity', 'bahdanau', 'transformer']:
            for classifier_name in ['linear', 'depthwise_separable_conv', 'communicating_conv']:

                print(bag_aggregator_name, class_connector_name, classifier_name)

                milnet = get_model(
                    bag_aggregator_name=bag_aggregator_name, class_connector_name=class_connector_name,
                    classifier_name=classifier_name, embedding_size=sample_input_size,
                    proj_size=sample_proj_size
                )
                prediction_bag, A, B, aux = milnet(input_embedding, sample_bag_lens)
                # print(f"prediction_bag: \n{prediction_bag}")
                # print("prediction_bag.shape:", prediction_bag.shape)
                # print("A.shape:", A.shape)
                # print("B.shape:", B.shape)

                assert prediction_bag.shape == (batch_size, sample_num_classes)
                assert A.shape == (batch_size, max_bag_length, sample_num_classes)
                assert B.shape == (batch_size, sample_num_classes, sample_input_size)

                if aux is not None:
                    # print("aux.shape:", aux.shape)
                    assert aux.shape == (batch_size, sample_num_classes)


                milnet_dict = milnet.state_dict()
                for i, name in enumerate(milnet_dict.keys()):
                    print(i, name, milnet_dict[name].shape)

                print("")

    print("All tests for shapes passed.")
