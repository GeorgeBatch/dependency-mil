"""
Usage example in: tests/unit/models/test_dsmil.py
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class FCLayer(nn.Module):
    """
    Fully-connected layer used to create the DSMIL model.

    Used as:
        i_classifier = mil.FCLayer(in_size=args.feats_size, out_size=args.num_classes).cuda()
        b_classifier = mil.BClassifier(input_size=args.feats_size, output_class=args.num_classes, v_dropout=args.dropout_node, q_nonlinear=args.non_linearity).cuda()
        milnet = mil.MILNet(i_classifier, b_classifier).cuda()

    After creation, we model can be initialized with pre-trained weights.
    """

    def __init__(self, in_size, out_size=1):
        super(FCLayer, self).__init__()
        self.fc = nn.Sequential(nn.Linear(in_size, out_size))

    def forward(self, feats):
        c = self.fc(feats)
        return feats, c


class IClassifier(nn.Module):
    """
    Used to compute features and predictions from images using weights pre-trained with simclr.

    In compute_feats.py it is used as:
        i_classifier = mil.IClassifier(resnet, num_feats, output_class=args.num_classes).cuda()

        # load pre-trained weights from simclr and prepare them for the i_classifier (preparation steps not shown here)
        i_classifier.load_state_dict(new_state_dict, strict=False)

        # used inside compute_feats(args, bags_list, i_classifier, feats_path, args.magnification)

        feats, classes = i_classifier(patches)
        feats = feats.cpu().numpy()
        feats = np.concatenate((feats, patch_locs), axis=1)
        feats_list.extend(feats)

        # feats_list is then used to create the dataframe of features, one row per patch
        # classes are not used at all here

    In testing_tcga.py and attention_map.py it is used as to create the DSMIL model with pre-trained weights that accepts a bag of patches from whole-slide images instead of pre-computed patch features.
        Compared to the FCLayer used for training on pre-computed patch features, this IClassifier has an additional feature_extractor that is used to extract features from the patches.
        First, the patch embedder weights are loaded in.
        Second, the aggregator weights are loaded in.
    """

    def __init__(self,
                 feature_extractor: nn.Module = nn.Identity(),
                 feature_size: int = 512,
                 output_class: int = 1
                 ) -> None:
        super(IClassifier, self).__init__()

        self.feature_extractor = feature_extractor
        self.fc = nn.Linear(feature_size, output_class)

    def forward(self, x):
        feats = self.feature_extractor(x)  # feats.shape = (bag_size, extracted_features)
        c = self.fc(feats.view(feats.shape[0], -1))  # c.shape = (bag_size, num_classes)
        return feats.view(feats.shape[0], -1), c


class BClassifier(nn.Module):
    def __init__(self,
                 input_size: int = 512,
                 output_class: int = 1,
                 q_size: int = 128,
                 q_nonlinear: bool = True,
                 v_size: int = 512,
                 v_dropout: float = 0.0,
                 v_identity: bool = True,
                 ):
        # extracted_features, L, bag_size
        super(BClassifier, self).__init__()

        if q_nonlinear:
            self.q = nn.Sequential(nn.Linear(input_size, q_size), nn.ReLU(), nn.Linear(q_size, q_size), nn.Tanh())
        else:
            self.q = nn.Linear(input_size, q_size)

        if v_identity:
            assert input_size == v_size, f"v_size={v_size} must be equal to input_size={input_size} when v_identity is True"
            self.v = nn.Identity()
        else:
            self.v = nn.Sequential(
                nn.Dropout(v_dropout),
                nn.Linear(input_size, v_size),
                nn.ReLU()
            )

        # 1D convolutional layer that can handle multiple classes (including binary)
        self.fcc = nn.Conv1d(in_channels=output_class, out_channels=output_class, kernel_size=v_size)
        # unlike the case with linear layer, we will have a different weights vector for each class
        # self.fcc_linear = nn.Linear(in_features=v_size, out_features=1)

    def forward(self, feats, c):  # feats.shape = (bag_size, extracted_features); c.shape = (bag_size, num_classes)
        device = feats.device
        V = self.v(feats)  # V.shape = (bag_size, v_size) - unsorted
        Q = self.q(feats).view(feats.shape[0], -1)  # Q.shape = (bag_size, q_size) - unsorted

        # compute attention scores for each instance and each class: A
        max_ins_prediction, max_ins_idx = torch.max(c,
                                                    dim=0)  # retrieve indices of largest class scores along the instance dimension, max_ins_idx.shape = (num_classes), but in this case num_classes is the number of critical instances (one critical instance per class)
        m_feats = feats[max_ins_idx, :]  # select critical instances: m_feats.shape = (num_classes, extracted_features)
        q_max = self.q(m_feats)  # compute queries of critical instances: q_max.shape = (num_classes, q_size)
        A = torch.mm(Q, q_max.transpose(0,
                                        1))  # compute inner product of Q to each entry of q_max: A.shape = (bag_size, num_classes) - each column contains unnormalized attention scores
        A = F.softmax(A / torch.sqrt(torch.tensor(Q.shape[1], dtype=torch.float32, device=device)),
                      0)  # normalize attention scores: A.shape = (bag_size, num_classes)

        # compute bag representation B
        B = torch.mm(A.transpose(0, 1), V)  # compute bag representation: B.shape = (num_classes, v_size)

        # compute bag prediction C from bag embedding B
        # C_alt = self.fcc_linear(B) # C_alt.shape: (num_classes, 1)
        C = self.fcc(B)  # C.shape: (num_classes, 1); 1 is left from the 1D convolutional layer
        C = C.squeeze(-1)  # C.shape: (num_classes, 1) -> (num_classes, )

        # reshape to pretend we have a batch size of 1 even thought we can not work with larger batch sizes
        A = A.unsqueeze(0)  # A.shape: (bag_size, num_classes) -> (1, bag_size, num_classes)
        B = B.unsqueeze(0)  # B.shape: (num_classes, v_size) -> (1, num_classes, v_size)
        C = C.unsqueeze(0)  # C.shape: (num_classes, ) -> (1, num_classes)
        max_ins_prediction = max_ins_prediction.unsqueeze(
            0)  # max_ins_prediction.shape: (num_classes, ) -> (1, num_classes)

        return C, A, B, max_ins_prediction


class MILNet(nn.Module):
    def __init__(self, i_classifier, b_classifier):
        """
        Parameters
        ----------
        i_classifier
        b_classifier
        """
        super(MILNet, self).__init__()
        self.i_classifier = i_classifier
        self.b_classifier = b_classifier

    def forward(self, x):
        """

        Parameters
        ----------
        x

        Returns
        -------
        tuple: prediction_bag, A, B, max_ins_prediction
        """
        # instance features and predictions
        feats, ins_predictions = self.i_classifier(x)

        # prediction_bag:       bag prediction                  (1, num_classes)
        # A:                    normalize attention scores      (bag_size, num_classes)
        # B:                    bag representation              (num_classes, v_size)
        # max_ins_prediction    critical instance prediction    (num_classes,)
        prediction_bag, A, B, max_ins_prediction = self.b_classifier(feats, ins_predictions)

        return prediction_bag, A, B, max_ins_prediction
