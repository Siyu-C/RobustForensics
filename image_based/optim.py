from torch.optim import SGD
from torch.optim import Adam
from torch.optim.optimizer import Optimizer, required

def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])
