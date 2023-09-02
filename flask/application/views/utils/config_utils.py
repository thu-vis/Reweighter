import sys
import os

class Config(object):
    def __init__(self):
        SERVER_ROOT = os.path.dirname(sys.modules[__name__].__file__)
        SERVER_ROOT = os.path.normpath(f'{SERVER_ROOT}/../..')
        print("SERVER_ROOT", SERVER_ROOT)
        self.root = SERVER_ROOT
        self.ORIGIN_DATA_DIR = {
            'medical': '/data/vica/medical/small_uniform_0.4/shuffled_x_train.npy',
            'cloth': '/data/vica/cloth/X-256.npy',
        }
        # self.dataname = 'medical115_121'
        # self.dataversion = 'case'
        self.dataname = 'cloth14'
        # self.dataname = 'medical'

config = Config()
