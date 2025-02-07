import torch.nn as nn


class LinearClassifier(nn.Module):
    def __init__(self, embedding_size: int = 512) -> None:
        super(LinearClassifier, self).__init__()
        # num_classes is a separate dimension, see comment in forward
        self.classifier = nn.Linear(embedding_size, 1)

    def forward(self, B):
        # B: (batch, num_classes, embedding_size) -> (batch, num_classes, 1).squeeze(-1) -> pred: (batch, num_classes)
        pred = self.classifier(B).squeeze(-1)
        return pred


class DSConvClassifier(nn.Module):
    """
    Setting groups=num_classes is equivalent to having a separate convolutional filter for each class - hence Depthwise Separable Convolution
    """

    def __init__(self, num_classes: int = 2, embedding_size: int = 512) -> None:
        super(DSConvClassifier, self).__init__()
        self.classifier = nn.Conv1d(
            in_channels=num_classes, out_channels=num_classes,
            kernel_size=embedding_size, groups=num_classes
        )

    def forward(self, B):
        # B: (batch, num_classes, embedding_size) -> pred: (batch, num_classes)
        pred = self.classifier(B).squeeze(-1)
        return pred


class CommunicatingConvClassifier(nn.Module):
    def __init__(self, num_classes: int = 2, embedding_size: int = 512) -> None:
        super(CommunicatingConvClassifier, self).__init__()
        self.classifier = nn.Conv1d(
            in_channels=num_classes, out_channels=num_classes, kernel_size=embedding_size)

    def forward(self, B):
        # B: (batch, num_classes, embedding_size) -> pred: (batch, num_classes)
        pred = self.classifier(B).squeeze(-1)
        return pred
