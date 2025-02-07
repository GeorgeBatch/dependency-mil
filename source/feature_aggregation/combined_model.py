import torch
import torch.nn as nn
import torch.nn.functional as F

# from source.constants import PROJECT_PATH
# # older code for building a model
# from source.feature_aggregation.bagplus_milnet import \
#     AbmilBagClassifier as AbmilBagClassifier_separate
# from source.feature_aggregation.bagplus_milnet import \
#     DsmilBagClassifier as DsmilBagClassifier_separate
# from source.feature_aggregation.bagplus_milnet import \
#     MILClassifier as MILClassifier_separate
# newer code for building a model
from source.feature_aggregation.class_connectors import BahdanauSelfAttention, TransformerSelfAttention
from source.feature_aggregation.classifier_heads import LinearClassifier, DSConvClassifier, CommunicatingConvClassifier
from source.feature_aggregation.instance_embedders import IdentityEmbedder, AdaptiveAvgPoolingEmbedder, LinearEmbedder, SliceEmbedder


class MILClassifier(nn.Module):
    def __init__(self,
                 instance_embedder: nn.Module = None,
                 instance_classifier: nn.Module = None,
                 bag_aggregator: nn.Module = None,
                 class_connector: nn.Module = None,
                 classifier: nn.Module = None
                 ) -> None:
        super(MILClassifier, self).__init__()

        self.instance_embedder = instance_embedder
        self.instance_classifier = instance_classifier
        self.bag_aggregator = bag_aggregator
        self.class_connector = class_connector
        self.classifier = classifier
        self.init_weights()

    def forward(self, x, bag_lens=None):
        device = next(self.parameters()).device
        x = x.to(device)

        instance_embeddings = self.compute_instance_embeds(x)
        instance_logits = self.instance_classifier(instance_embeddings) if self.instance_classifier is not None else None
        # squeeze the last dimension of instance_logits
        #   if binary classification, the last dimension is 1 so we can remove it
        #   if multi-class classification, the last dimension is the number of classes, so it will not be squeezed
        instance_logits = instance_logits.squeeze(dim=-1) if instance_logits is not None else None
        att_bag, feat_bag, aux = self.bag_aggregator(instance_embeddings, bag_lens)
        feat_bag = self.class_connector(
            feat_bag)  # feat_bag: (batch, num_classes, embedding_size) -> feat_bag: (batch, num_classes, embedding_size)
        pred_bag = self.classifier(
            feat_bag)  # feat_bag: (batch, num_classes, embedding_size) -> pred_bag: (batch, num_classes)

        return pred_bag, att_bag, feat_bag, aux, instance_logits

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
                 ) -> None:
        super(AbmilBagClassifier, self).__init__()

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
        if not self.gated:
            A = self.attention(x)  # batch x max_bag_length x num_classes
        else:
            A_v = self.attention_v(x)  # batch x max_bag_length x proj_size
            A_u = self.attention_u(x)  # batch x max_bag_length x proj_size
            A = self.attention(A_v * A_u)  # batch x max_bag_length x num_classes

        mask = torch.zeros((x.shape[0], x.shape[1], A.shape[-1]), device=device, dtype=torch.bool)  # batch x bag x 1
        for i in range(mask.shape[0]):
            mask[i, :bag_lens[i], :] = True
        A[~mask] = float('-inf')  # apply mask
        A = F.softmax(A, 1)  # over instances in a bag, batch x bag x num_classes
        B = torch.bmm(torch.permute(A, (0, 2, 1)), x)  # batch x num_classes x embedding_size

        return A, B, None


class DsmilBagClassifier(nn.Module):
    def __init__(self,
                 num_classes: int = 1,
                 embedding_size: int = 512,
                 q_size: int = 128,
                 q_nonlinear: bool = True,
                 v_size: int = 512,
                 v_identity: bool = True,
                 ) -> None:
        super(DsmilBagClassifier, self).__init__()

        if q_nonlinear:
            self.q_net = nn.Sequential(
                nn.Linear(embedding_size, q_size),
                nn.ReLU(),
                nn.Linear(q_size, q_size),
                nn.Tanh()
            )
            # Code from BagPlus repo:
            # self.q_net = nn.Sequential(nn.Linear(embedding_size, embedding_size), nn.PReLU(), nn.Linear(embedding_size, q_size),  nn.Tanh())
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

        V = self.v_net(x)  # batch x max_bag_length x v_size
        Q = self.q_net(x)  # batch x max_bag_length x q_size

        Inst = self.instance_classifier(x)  # batch x max_bag_length x num_classes
        for i in range(Inst.shape[0]):
            Inst[i, bag_lens[i]:, :] = float(
                '-inf')  # instanses that are pure padding should never be considered as critical
        I_m, M_id = torch.max(Inst, dim=1)  # batch x num_classes

        Q_m = torch.gather(Q, 1, M_id.unsqueeze(2).expand(M_id.size(0), M_id.size(1),
                                                          Q.size(2)))  # batch x num_classes x q_size

        A = torch.bmm(Q, torch.permute(Q_m, (0, 2, 1)))  # batch x max_bag_length x num_classes

        real_instances_mask = torch.zeros((x.shape[0], x.shape[1], A.shape[-1]), device=device,
                                          dtype=torch.bool)  # batch x bag x 1
        for i in range(real_instances_mask.shape[0]):
            real_instances_mask[i, :bag_lens[i], :] = True
        A[~real_instances_mask] = float('-inf')  # apply mask to all purely padded instances

        A = F.softmax(A, 1)  # over instances in a bag, batch x max_bag_length x num_classes
        # TODO: check what happens if we normalize A by dividing by sqrt(q_size)
        # sqrt_q_size = torch.sqrt(torch.tensor(self.q_size, device=device, dtype=torch.float32))
        # A = F.softmax(A / sqrt_q_size, 1) # over instances in a bag, batch x max_bag_length x num_classes

        B = torch.bmm(torch.permute(A, (0, 2, 1)), V)  # batch x num_classes x v_size
        return A, B, I_m


# ------------------ Class Connectors ---------------------
# Imported: `from source.feature_aggregation.class_connectors import BahdanauSelfAttention, TransformerSelfAttention`


# ------------------ Classifier Heads ---------------------
# Imported: `from source.feature_aggregation.classifier_heads import LinearClassifier, DSConvClassifier, CommunicatingConvClassifier`


# ------------------ How it works together --------------------

def get_model(
        instance_embedder_name: str = "identity",
        instance_classifier_name: str = None,
        bag_aggregator_name: str = "dsmil",
        aggregator_kwargs: dict = None,
        class_connector_name: str = "identity",
        classifier_name: str = "communicating_conv",
        num_classes: int = 2,
        feats_size: int = 512,
        instance_embedder_output_size: int = 512,
) -> nn.Module:
    assert instance_embedder_name in ["identity", "adaptive_avg_pooling", "linear", "slice_first_outputsize_features"], \
        f"instance_embedder_name={instance_embedder_name} is not supported"
    assert instance_classifier_name in [None, "linear"], f"instance_classifier_name={instance_classifier_name} is not supported"
    assert bag_aggregator_name in ["abmil", "dsmil"], f"bag_aggregator_name={bag_aggregator_name} is not supported"
    assert class_connector_name in ["identity", "bahdanau", "transformer"], \
        f"class_connector_name={class_connector_name} is not supported"
    assert classifier_name in ["linear", "depthwise_separable_conv", "communicating_conv"], \
        f"classifier_name={classifier_name} is not supported"

    if instance_embedder_name == "identity":
        assert feats_size == instance_embedder_output_size, \
            f"For identity instance embedder, feats_size ({feats_size}) must be equal to instance_embedder_output_size ({instance_embedder_output_size})"
        instance_embedder = IdentityEmbedder()
    elif instance_embedder_name == "adaptive_avg_pooling":
        assert feats_size > instance_embedder_output_size, \
            f"For adaptive_avg_pooling instance embedder, feats_size ({feats_size}) must be greater than instance_embedder_output_size ({instance_embedder_output_size})"
        instance_embedder = AdaptiveAvgPoolingEmbedder(instance_embedder_output_size)
    elif instance_embedder_name == "linear":
        instance_embedder = LinearEmbedder(feats_size, instance_embedder_output_size)
    elif instance_embedder_name == "slice_first_outputsize_features":
        instance_embedder = SliceEmbedder(instance_embedder_output_size)
    else:
        raise NotImplementedError(f"instance_embedder_name={instance_embedder_name} is not supported")

    if instance_classifier_name is None:
        instance_classifier = None
    elif instance_classifier_name == "linear":
        instance_classifier = nn.Linear(instance_embedder_output_size, 1)
    else:
        raise NotImplementedError(f"instance_classifier_name={instance_classifier_name} is not supported")

    if bag_aggregator_name == "abmil":
        bag_aggregator = AbmilBagClassifier(
            num_classes=num_classes,
            embedding_size=instance_embedder_output_size,
            **aggregator_kwargs, # proj_size=128, gated=False
        )
        bag_aggregator_output_embedding_size = instance_embedder_output_size
    elif bag_aggregator_name == "dsmil":
        bag_aggregator = DsmilBagClassifier(
            num_classes=num_classes,
            embedding_size=instance_embedder_output_size,
            **aggregator_kwargs,  # v_size = instance_embedder_output_size, v_identity=True, q_size=128, q_nonlinear=True
        )
        # get the size from comments in DSMIL
        bag_aggregator_output_embedding_size = aggregator_kwargs["v_size"]
    else:
        raise NotImplementedError(f"bag_aggregator_name={bag_aggregator_name} is not supported")

    if class_connector_name == "identity":
        class_connector = nn.Identity()
    elif class_connector_name == "bahdanau":
        class_connector = BahdanauSelfAttention(embedding_size=bag_aggregator_output_embedding_size)
    elif class_connector_name == "transformer":
        class_connector = TransformerSelfAttention(embedding_size=bag_aggregator_output_embedding_size, num_heads=1)
    else:
        raise NotImplementedError(f"class_connector_name={class_connector_name} is not supported")

    if classifier_name == "linear":
        classifier = LinearClassifier(embedding_size=instance_embedder_output_size)
    elif classifier_name == "depthwise_separable_conv":
        classifier = DSConvClassifier(num_classes=num_classes, embedding_size=instance_embedder_output_size)
    elif classifier_name == "communicating_conv":
        classifier = CommunicatingConvClassifier(num_classes=num_classes, embedding_size=instance_embedder_output_size)
    else:
        raise NotImplementedError(f"classifier_name={classifier_name} is not supported")

    milnet = MILClassifier(instance_embedder, instance_classifier, bag_aggregator, class_connector, classifier)

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

                # ----------------------
                # milnet_separate

                # embedder = torch.nn.Identity()
                # if bag_aggregator_name == 'dsmil':
                #     aggregator = DsmilBagClassifier_separate(
                #         v_identity=True,
                #         num_classes=sample_num_classes,
                #         embedding_size=sample_input_size,
                #         q_size=sample_proj_size,
                #         v_size=sample_input_size,
                #         class_connector_name=class_connector_name,
                #         classifier_name=classifier_name,
                #     )
                # elif bag_aggregator_name == 'abmil':
                #     aggregator = AbmilBagClassifier_separate(
                #         gated=False,
                #         num_classes=sample_num_classes,
                #         embedding_size=sample_input_size,
                #         proj_size=sample_proj_size,
                #         class_connector_name=class_connector_name,
                #         classifier_name=classifier_name,
                #     )
                # else:
                #     raise NotImplementedError(f"bag_aggregator_name={bag_aggregator_name} is not supported")
                # milnet_separate = MILClassifier_separate(instance_embedder=embedder, classifier_bag=aggregator)
                # ----------------------

                # ----------------------
                # dict visual comparison

                milnet_dict = milnet.state_dict()
                for i, name in enumerate(milnet_dict.keys()):
                    print(i, name, milnet_dict[name].shape)

                # milnet_separate_dict = milnet_separate.state_dict()
                # for i, name in enumerate(milnet_separate_dict.keys()):
                #     print(i, name, milnet_separate_dict[name].shape)
                # ----------------------

                # ----------------------
                # new separate dict creation

                # milnet_separate_new_dict = {}
                # for i, (nameA, nameB) in enumerate(
                #         zip(
                #             milnet_dict.keys(),
                #             milnet_separate_dict.keys()
                #         )):
                #     # print(i, '\n\t', nameA, '\n\t', nameB)
                #     assert milnet_dict[nameA].shape == milnet_separate_dict[nameB].shape
                #     assert ".".join(nameA.split(".")[-2:]) == ".".join(nameB.split(".")[-2:])
                #     layer_correspondence_dict[nameB] = nameA
                #     milnet_separate_new_dict[nameB] = milnet_dict[nameA]

                # milnet_separate.load_state_dict(milnet_separate_new_dict, strict=True)

                # ----------------------
                # verification the two models give the same outputs

                # for tensor_a, tensor_b in zip(
                #         milnet_separate(input_embedding, sample_bag_lens),
                #         milnet(input_embedding, sample_bag_lens)
                # ):
                #     if tensor_a is None:
                #         assert tensor_b is None
                #     else:
                #         assert torch.allclose(tensor_a, tensor_b, atol=1e-8)

                # with open(
                #         os.path.join(
                #             PROJECT_PATH,
                #             f"weights/old2new-layer-names/{bag_aggregator_name}-{class_connector_name}-{classifier_name}.json"
                #         ), 'w') as f:
                #     json.dump(layer_correspondence_dict, f, indent="\t")
                # ----------------------

                print("")

    print("All tests for shapes passed.")
