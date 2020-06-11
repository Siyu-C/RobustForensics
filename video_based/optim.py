from torch.optim import SGD
from torch.optim import Adam

        
def optim_entry(config):
    return globals()[config['type']](**config['kwargs'])
