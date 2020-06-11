from .slowfast import *

def model_entry(config):
    return globals()[config['arch']](**config['kwargs'])
