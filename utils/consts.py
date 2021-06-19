import torch
from json import JSONEncoder

class Consts:
    def __init__(self):
        self.USE_CUDA = True if torch.cuda.is_available() else False
        if self.USE_CUDA:
            print('Running with GPU...')
        else:
            print('Running with CPU...')

        self.DEVICE = "cuda" if self.USE_CUDA else "cpu"

        self.DATASET = "cifar-10"
        self.NUM_CLASSES = 10

        # self.MODEL = "VGG11" # VGG has too many maxpool layers. Don't use it on Cifar-10!
        self.MODEL = "ResNet18"
        # self.MODEL = "ResNeXt29_2x64d"
        # self.MODEL = "PreActResNet18"
        # self.MODEL = "GoogLeNet"
        # self.MODEL = "EfficientNetB0"
        # self.MODEL = "MobileNetV2"
        # self.MODEL = "ShuffleNetV2"

        self.ADV_TRAIN = False   # Avdersarial training or not
        self.EPS = 0.3          # Total epsilon for FGM and PGD attacks.

        self.RANDOM_SEED = 0
        self.BATCH_SIZE = 128
        self.NUM_WORKERS = 2
        self.NUM_EPOCHS = 200
        self.LEARNING_RATE = 1e-3

        # self.OPTIMIZER = "Adam"
        self.OPTIMIZER = "SGD"
        self.MOMENTUM = 0.9
        self.WEIGHT_DECAY = 4e-5

        # self.AUXILIARY_TYPE = "Recon"
        self.AUXILIARY_TYPE = "Fourier"
        # self.AUXILIARY_TYPE = None
        self.AUXILIARY_WEIGHT = 0.005

class ClassEncoder(JSONEncoder):
    def default(self, o):
        return o.__dict__
