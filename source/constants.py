from os import path as osp

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../'))

CONFIGS_PATH = osp.join(PROJECT_PATH, 'configs')
EXPERIMENTS_PATH = osp.join(PROJECT_PATH, 'experiments')

EXTRACTOR_NAMES_2_WEIGHTS_PATHS = {
    'simclr-tcga-lung_resnet18-10x': osp.join(
        PROJECT_PATH, 'weights-pre-trained/simclr-tcga-lung/weights-10x/model-v1.pth'),
    'simclr-tcga-lung_resnet18-2.5x': osp.join(
        PROJECT_PATH, 'weights-pre-trained/simclr-tcga-lung/weights-2.5x/model-v1.pth'),
    'simclr-camelyon16_resnet18-20x': osp.join(
        PROJECT_PATH, 'weights-pre-trained/simclr-camelyon16/weights-20x/model-v2.pth'),
    'simclr-camelyon16_resnet18-5x': osp.join(
        PROJECT_PATH, 'weights-pre-trained/simclr-camelyon16/weights-5x/model.pth'),
}

DATASET_SPECIFIC_NORMALIZATION_CONSTANTS_PATH = osp.join(PROJECT_PATH, 'source/feature_extraction/img_normalisation_constants.json')
ALL_IMG_NORMS = (
    'imagenet',
    'openai_clip',
    'uniform',
    'resize_only',
)
