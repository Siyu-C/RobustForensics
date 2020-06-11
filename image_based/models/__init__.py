from .models import model_selection
from .efficientnet_pytorch.model import *
from .resnet import *

def model_entry(config):
    if config['arch'] == 'xception':
        model = model_selection(
            modelname='xception',
            num_out_classes=config['kwargs']['num_classes'],
            pretrain_path=config['kwargs'].get('pretrain_path', None)
        )
        return model
    elif config['arch'].startswith('efficientnet'):
        model = EfficientNet.from_pretrained(config['arch'], **config['kwargs'])
        return model
    else:
        return globals()[config['arch']](**config['kwargs'])
