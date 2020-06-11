import os

import torch.distributed as dist
import torch
import torch.nn as nn
from .xception import xception
from utils import print_with_rank


def pretrain(model, state_dict):
    own_state = model.state_dict()
    for name, param in state_dict.items():
        if name in own_state:
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print_with_rank('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print_with_rank("But don't worry about it. Continue pretraining.")


def return_pytorch04_xception(pretrain_path='../pretrain/xception-b5690688.pth', **kwargs):
    # Raises warning "src not broadcastable to dst" but thats fine
    model = xception(num_classes=2)
    if pretrain_path is not None:
        state_dict = torch.load(pretrain_path)
        for name, weights in state_dict.items():
            if 'pointwise' in name:
                state_dict[name] = weights.unsqueeze(-1).unsqueeze(-1)
        ckpt_keys = set(state_dict.keys())
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - ckpt_keys
        for k in missing_keys:
            print_with_rank('caution: missing keys from checkpoint {}'.format(k))
        pretrain(model, state_dict)
    return model


class TransferModel(nn.Module):
    """
    Simple transfer learning model that takes an possibly imagenet pretrained model with
    a fc layer as base model and retrains a new fc layer for num_out_classes
    """
    def __init__(self, modelchoice, num_out_classes=2, pretrain_path=None, dropout=None):
        super(TransferModel, self).__init__()
        self.modelchoice = modelchoice
        if modelchoice == 'xception':
            self.model = return_pytorch04_xception(pretrain_path=pretrain_path)
            # Replace fc
            num_ftrs = self.model.fc.in_features
            if not dropout:
                self.model.fc = nn.Linear(num_ftrs, num_out_classes)
            else:
                print('Using dropout', dropout)
                self.model.fc = nn.Sequential(
                    nn.Dropout(p=dropout),
                    nn.Linear(num_ftrs, num_out_classes)
                )
        else:
            raise Exception('Choose valid model, e.g. resnet50')

    def forward(self, x):
        x = self.model(x)
        return x


def model_selection(modelname, num_out_classes=2, pretrain_path=None):
    """
    :param modelname, num_out_classes, pretrained, dropout:
    :return: model, image size
    """
    return TransferModel(modelchoice=modelname,
                         num_out_classes=num_out_classes,
                         pretrain_path=pretrain_path)
