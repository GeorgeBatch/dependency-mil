from os import path as osp

PROJECT_PATH = osp.abspath(osp.join(osp.dirname(osp.realpath(__file__)), '../'))

CONFIGS_PATH = osp.join(PROJECT_PATH, 'configs')
EXPERIMENTS_PATH = osp.join(PROJECT_PATH, 'experiments')
