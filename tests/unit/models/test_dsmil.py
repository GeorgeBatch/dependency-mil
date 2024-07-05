import torch
from torchvision.models import resnet18

from source.models.dsmil import BClassifier, FCLayer, IClassifier, MILNet

print("This is the MIL model.")

sample_batch_size = 1  # only works for batch-size=1 right now
sample_bag_length = 10
sample_input_size = 512
sample_output_class = 2


def test_dsmil_from_embedding():
    torch.manual_seed(0)
    input_embedding = torch.rand((sample_bag_length, sample_input_size), dtype=torch.float32)

    i_classifier = FCLayer(in_size=sample_input_size, out_size=sample_output_class)
    b_classifier = BClassifier(
        input_size=sample_input_size,
        output_class=sample_output_class,
    )
    milnet = MILNet(i_classifier, b_classifier)

    prediction_bag, A, B, max_ins_prediction = milnet(input_embedding)
    assert prediction_bag.shape == (sample_batch_size, sample_output_class)
    assert A.shape == (sample_batch_size, sample_bag_length, sample_output_class)
    assert B.shape == (sample_batch_size, sample_output_class, sample_input_size)
    assert max_ins_prediction.shape == (sample_batch_size, sample_output_class)


def test_dsmil_from_picture():
    torch.manual_seed(0)
    input_picture_batch = torch.rand((sample_bag_length, 3, 224, 224), dtype=torch.float32)

    resnet = resnet18()
    resnet.fc = torch.nn.Identity()
    i_classifier = IClassifier(feature_extractor=resnet, feature_size=sample_input_size,
                               output_class=sample_output_class)
    b_classifier = BClassifier(input_size=sample_input_size, output_class=sample_output_class)
    milnet = MILNet(i_classifier, b_classifier)

    prediction_bag, A, B, max_ins_prediction = milnet(input_picture_batch)

    assert prediction_bag.shape == (sample_batch_size, sample_output_class)
    assert A.shape == (sample_batch_size, sample_bag_length, sample_output_class)
    assert B.shape == (sample_batch_size, sample_output_class, sample_input_size)
    assert max_ins_prediction.shape == (sample_batch_size, sample_output_class)
