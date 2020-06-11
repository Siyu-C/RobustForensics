from __future__ import print_function
import torch
import torch.backends.cudnn as cudnn
import numpy as np
import cv2
import os
import pandas as pd
import copy
import time
import torchvision

from face_detect_lib.models.retinaface import RetinaFace
from face_detect_lib.layers.functions.prior_box import PriorBox
from face_detect_lib.utils.box_utils import decode_batch, decode_landm_batch, decode, decode_landm

global VIDEO_LOAD_TIME, DETECT_FACE_TIME, VIDEO_TIME
VIDEO_LOAD_TIME, DETECT_FACE_TIME, VIDEO_TIME = 0, 0, 0

cfg_mnet = {
    'name': 'mobilenet0.25',
    'min_sizes': [[16, 32], [64, 128], [256, 512]],
    'steps': [8, 16, 32],
    'variance': [0.1, 0.2],
    'clip': False,
    'loc_weight': 2.0,
    'gpu_train': True,
    'batch_size': 32,
    'ngpu': 1,
    'epoch': 250,
    'decay1': 190,
    'decay2': 220,
    'image_size': 640,
    'pretrain': True,
    'return_layers': {'stage1': 1, 'stage2': 2, 'stage3': 3},
    'in_channel': 32,
    'out_channel': 64
}


class Config:
    def __init__(self):
        self.cuda = True
        self.face_model_path = './face_detect_ckpt/mobilenet0.25_Final.pth'
        self.model_name = 'mobile0.25'
        self.origin_size = False
        self.confidence_threshold = 0.02
        self.top_k = 5000
        self.nms_threshold = 0.4
        self.keep_top_k = 750
        self.target_size = 400
        self.max_size = 2150
        self.model_cfg = cfg_mnet
        self.vis_thres = 0.8


pipeline_cfg = Config()


def check_keys(model, pretrained_state_dict):
    ckpt_keys = set(pretrained_state_dict.keys())
    model_keys = set(model.state_dict().keys())
    used_pretrained_keys = model_keys & ckpt_keys
    unused_pretrained_keys = ckpt_keys - model_keys
    missing_keys = model_keys - ckpt_keys
    print('Missing keys:{}'.format(len(missing_keys)))
    print('Unused checkpoint keys:{}'.format(len(unused_pretrained_keys)))
    print('Used keys:{}'.format(len(used_pretrained_keys)))
    assert len(used_pretrained_keys) > 0, 'load NONE from pretrained checkpoint'
    return True


def remove_prefix(state_dict, prefix):
    ''' Old style model is stored with all names of parameters sharing common prefix 'module.' '''
    print('remove prefix \'{}\''.format(prefix))
    f = lambda x: x.split(prefix, 1)[-1] if x.startswith(prefix) else x
    return {f(key): value for key, value in state_dict.items()}


def load_model(model, pretrained_path, load_to_cpu):
    print('Loading pretrained model from {}'.format(pretrained_path))
    load_to_cpu = not load_to_cpu
    if load_to_cpu:
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage)
    else:
        print('using cuda!')
        device = torch.cuda.current_device()
        pretrained_dict = torch.load(pretrained_path, map_location=lambda storage, loc: storage.cuda(device))
    if "state_dict" in pretrained_dict.keys():
        pretrained_dict = remove_prefix(pretrained_dict['state_dict'], 'module.')
    else:
        pretrained_dict = remove_prefix(pretrained_dict, 'module.')
    check_keys(model, pretrained_dict)
    model.load_state_dict(pretrained_dict, strict=False)
    return model


def py_cpu_nms(dets, thresh):
    """Pure Python NMS baseline."""
    x1 = dets[:, 0]
    y1 = dets[:, 1]
    x2 = dets[:, 2]
    y2 = dets[:, 3]
    scores = dets[:, 4]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]
    keep = []
    while order.size > 0:
        i = order[0]
        keep.append(i)
        xx1 = np.maximum(x1[i], x1[order[1:]])
        yy1 = np.maximum(y1[i], y1[order[1:]])
        xx2 = np.minimum(x2[i], x2[order[1:]])
        yy2 = np.minimum(y2[i], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h
        ovr = inter / (areas[i] + areas[order[1:]] - inter)

        inds = np.where(ovr <= thresh)[0]
        order = order[inds + 1]
    return keep


def get_boundingbox(face, width, height, scale=1.3, minsize=None):
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    size_bb = int(max(x2 - x1, y2 - y1) * scale)
    if minsize:
        if size_bb < minsize:
            size_bb = minsize
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2
    x1 = max(int(center_x - size_bb // 2), 0)
    y1 = max(int(center_y - size_bb // 2), 0)
    size_bb = min(width - x1, size_bb)
    size_bb = min(height - y1, size_bb)
    return x1, y1, size_bb


def adjust_boundingbox(face, width, height, size):
    """
    :param face: dlib face class
    :param width: frame width
    :param height: frame height
    :param size: set bounding box size
    :return: x, y, bounding_box_size in opencv form
    """
    x1 = face[0]
    y1 = face[1]
    x2 = face[2]
    y2 = face[3]
    center_x, center_y = (x1 + x2) // 2, (y1 + y2) // 2

    # Check for out of bounds, x-y top left corner
    x1 = min(max(int(center_x - size // 2), 0), width - size)
    y1 = min(max(int(center_y - size // 2), 0), height - size)

    return [x1, y1, size]


def detect_face(img_list, detect_record):
    detect_face_begin = time.time()
    im_shape = img_list[0].shape
    detect_key = str(im_shape[0]) + '*' + str(im_shape[1])
    if detect_key not in detect_record:
        print(detect_key + ' not in dict')
        im_size_min = np.min(im_shape[0:2])
        im_size_max = np.max(im_shape[0:2])
        resize = float(pipeline_cfg.target_size) / float(im_size_min)
        # prevent bigger axis from being more than max_size:
        if np.round(resize * im_size_max) > pipeline_cfg.max_size:
            resize = float(pipeline_cfg.max_size) / float(im_size_max)
        im_height, im_width = int(im_shape[0] * resize), int(im_shape[1] * resize)
        detect_record[detect_key] = {'resize': resize, 'resized_h': im_height, 'resized_w': im_width}
        priorbox = PriorBox(pipeline_cfg.model_cfg, image_size=(im_height, im_width))
        priors = priorbox.forward()
        priors = priors.to(pipeline_cfg.device)
        detect_record[detect_key]['priors'] = priors

    # detect face
    detect_info = detect_record[detect_key]
    resize = detect_info['resize']
    resize_img_list = []
    result_dets_list = []
    batch_size = 8
    detect_nms_time = 0
    for img_idx, img in enumerate(img_list):
        if detect_info['resize'] != 1:
            img = cv2.resize(img, None, None, fx=detect_info['resize'], fy=detect_info['resize'],
                             interpolation=cv2.INTER_LINEAR)
            img = np.float32(img)
        else:
            img = np.float32(img)

        resize_img_list.append(img)
        img_idx += 1
        if img_idx % batch_size == 0 or img_idx == len(img_list):
            im_height, im_width, _ = resize_img_list[0].shape
            scale = torch.Tensor([resize_img_list[0].shape[1], resize_img_list[0].shape[0], resize_img_list[0].shape[1],
                                  resize_img_list[0].shape[0]])
            resize_img_list = np.stack(resize_img_list, axis=0)  # [n,h,w,c]
            resize_img_list -= (104, 117, 123)
            resize_img_list = resize_img_list.transpose(0, 3, 1, 2)
            resize_img_list = torch.from_numpy(resize_img_list)
            resize_img_list = resize_img_list.to(pipeline_cfg.device)
            scale = scale.to(pipeline_cfg.device)
            loc, conf, landms = pipeline_cfg.net(resize_img_list)
            priors = detect_info['priors']
            prior_data = priors.data
            boxes = decode_batch(loc.data, prior_data, pipeline_cfg.model_cfg['variance'])
            boxes = boxes * scale / resize  # [batchsize, proposals, 4]
            scores = conf[:, :, 1]  # [batchsize, proposals]


            detect_nms_begin = 0
            for per_idx in range(boxes.shape[0]):
                box, score = boxes[per_idx, :, :], scores[per_idx, :]
                inds = torch.nonzero(score > pipeline_cfg.confidence_threshold)[:,0]
                box, score = box[inds, :], score[inds]
                dets = torch.cat((box, score[:, None]), dim=1)
                keep = torchvision.ops.nms(box, score, pipeline_cfg.nms_threshold)
                dets = dets[keep, :]
                dets = dets.data.cpu().numpy()
                result_dets_list.append(dets)
            resize_img_list = []
            detect_nms_end = 0
            detect_nms_time += detect_nms_end - detect_nms_begin
    return result_dets_list


def vanilla_bbox_iou_overlaps(b1, b2):
    """
    Arguments:
        b1: dts, [n, >=4] (x1, y1, x2, y2, ...)
        b1: gts, [n, >=4] (x1, y1, x2, y2, ...)

    Returns:
        intersection-over-union pair-wise.
    """
    area1 = (b1[:, 2] - b1[:, 0] + 1) * (b1[:, 3] - b1[:, 1] + 1)
    area2 = (b2[:, 2] - b2[:, 0] + 1) * (b2[:, 3] - b2[:, 1] + 1)
    lt = np.maximum(b1[:, np.newaxis, :2], b2[np.newaxis, :, :2])
    rb = np.minimum(b1[:, np.newaxis, 2:4], b2[np.newaxis, :, 2:4])
    wh = np.maximum(rb - lt + 1, 0.)
    inter_area = wh[:, :, 0] * wh[:, :, 1]
    union_area = area1[:, np.newaxis] + area2[np.newaxis, :] - inter_area
    return inter_area / np.maximum(union_area, 1.)


def init_face_detecor():
    torch.set_grad_enabled(False)
    pipeline_cfg.net = RetinaFace(cfg=pipeline_cfg.model_cfg, phase='test')
    pipeline_cfg.net = load_model(pipeline_cfg.net, pipeline_cfg.face_model_path, pipeline_cfg.cuda)
    pipeline_cfg.net.eval()
    cudnn.benchmark = True
    pipeline_cfg.device = torch.device("cuda" if pipeline_cfg.cuda else "cpu")
    pipeline_cfg.net = pipeline_cfg.net.to(pipeline_cfg.device)
    return pipeline_cfg.net


# definition of slowfast model
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable
from functools import partial


def conv3x3x3(in_planes, out_planes, stride=1):
    # 3x3x3 convolution with padding
    return nn.Conv3d(
        in_planes,
        out_planes,
        kernel_size=3,
        stride=stride,
        padding=1,
        bias=False)


def downsample_basic_block(x, planes, stride):
    out = F.avg_pool3d(x, kernel_size=1, stride=stride)
    zero_pads = torch.Tensor(
        out.size(0), planes - out.size(1), out.size(2), out.size(3),
        out.size(4)).zero_()
    if isinstance(out.data, torch.cuda.FloatTensor):
        zero_pads = zero_pads.cuda()

    out = Variable(torch.cat([out.data, zero_pads], dim=1))

    return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None, head_conv=1):
        super(Bottleneck, self).__init__()
        if head_conv == 1:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=1, bias=False)
            self.bn1 = nn.BatchNorm3d(planes)
        elif head_conv == 3:
            self.conv1 = nn.Conv3d(inplanes, planes, kernel_size=(3, 1, 1), bias=False, padding=(1, 0, 0))
            self.bn1 = nn.BatchNorm3d(planes)
        else:
            raise ValueError("Unsupported head_conv!")
        self.conv2 = nn.Conv3d(
            planes, planes, kernel_size=(1, 3, 3), stride=(1, stride, stride), padding=(0, 1, 1), bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.conv3 = nn.Conv3d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm3d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        if downsample is not None:
            self.downsample_bn = nn.BatchNorm3d(planes * 4)
        self.stride = stride

    def forward(self, x):
        res = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            res = self.downsample(x)
            res = self.downsample_bn(res)

        out = out + res
        out = self.relu(out)

        return out


class SlowFast(nn.Module):
    def __init__(self, block=Bottleneck, layers=[3, 4, 6, 3], num_classes=400, shortcut_type='B',
                 dropout=0.5, alpha=8, beta=0.125, tau=16, zero_init_residual=False):
        super(SlowFast, self).__init__()

        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        '''Fast Network'''
        self.fast_inplanes = int(64 * beta)
        fast_inplanes = self.fast_inplanes
        self.fast_conv1 = nn.Conv3d(3, fast_inplanes, kernel_size=(5, 7, 7), stride=(1, 2, 2), padding=(2, 3, 3),
                                    bias=False)
        self.fast_bn1 = nn.BatchNorm3d(int(64 * beta))
        self.fast_relu = nn.ReLU(inplace=True)
        self.fast_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.fast_res1 = self._make_layer_fast(
            block, int(64 * beta), layers[0], shortcut_type, head_conv=3)
        self.fast_res2 = self._make_layer_fast(
            block, int(128 * beta), layers[1], shortcut_type, stride=2, head_conv=3)
        self.fast_res3 = self._make_layer_fast(
            block, int(256 * beta), layers[2], shortcut_type, stride=2, head_conv=3)
        self.fast_res4 = self._make_layer_fast(
            block, int(512 * beta), layers[3], shortcut_type, stride=2, head_conv=3)

        '''Slow Network'''
        self.slow_inplanes = 64
        slow_inplanes = self.slow_inplanes
        self.slow_conv1 = nn.Conv3d(3, slow_inplanes, kernel_size=(1, 7, 7), stride=(1, 2, 2), padding=(0, 3, 3),
                                    bias=False)
        self.slow_bn1 = nn.BatchNorm3d(64)
        self.slow_relu = nn.ReLU(inplace=True)
        self.slow_maxpool = nn.MaxPool3d(kernel_size=(1, 3, 3), stride=(1, 2, 2), padding=(0, 1, 1))

        self.slow_res1 = self._make_layer_slow(
            block, 64, layers[0], shortcut_type, head_conv=1)
        self.slow_res2 = self._make_layer_slow(
            block, 128, layers[1], shortcut_type, stride=2, head_conv=1)
        self.slow_res3 = self._make_layer_slow(
            block, 256, layers[2], shortcut_type, stride=2, head_conv=3)  # Here we add non-degenerate t-conv
        self.slow_res4 = self._make_layer_slow(
            block, 512, layers[3], shortcut_type, stride=2, head_conv=3)  # Here we add non-degenerate t-conv

        '''Lateral Connections'''
        self.Tconv1 = nn.Conv3d(int(64 * beta), int(128 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv2 = nn.Conv3d(int(256 * beta), int(512 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv3 = nn.Conv3d(int(512 * beta), int(1024 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)
        self.Tconv4 = nn.Conv3d(int(1024 * beta), int(2048 * beta), kernel_size=(5, 1, 1), stride=(alpha, 1, 1),
                                padding=(2, 0, 0), bias=False)

        self.dp = nn.Dropout(dropout)
        self.fc = nn.Linear(self.fast_inplanes + self.slow_inplanes, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv3d):
                m.weight = nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.BatchNorm3d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)

    def forward(self, input):
        fast, Tc = self.FastPath(input)
        slow_stride = self.alpha
        slow = self.SlowPath(input[:, :, ::slow_stride, :, :], Tc)

        x = torch.cat([slow, fast], dim=1)
        x = self.dp(x)
        x = self.fc(x)
        return x

    def SlowPath(self, input, Tc):
        x = self.slow_conv1(input)
        x = self.slow_bn1(x)
        x = self.slow_relu(x)
        x = self.slow_maxpool(x)
        x = torch.cat([x, Tc[0]], dim=1)
        x = self.slow_res1(x)
        x = torch.cat([x, Tc[1]], dim=1)
        x = self.slow_res2(x)
        x = torch.cat([x, Tc[2]], dim=1)
        x = self.slow_res3(x)
        x = torch.cat([x, Tc[3]], dim=1)
        x = self.slow_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x

    def FastPath(self, input):
        x = self.fast_conv1(input)
        x = self.fast_bn1(x)
        x = self.fast_relu(x)
        x = self.fast_maxpool(x)
        Tc1 = self.Tconv1(x)
        x = self.fast_res1(x)
        Tc2 = self.Tconv2(x)
        x = self.fast_res2(x)
        Tc3 = self.Tconv3(x)
        x = self.fast_res3(x)
        Tc4 = self.Tconv4(x)
        x = self.fast_res4(x)
        x = nn.AdaptiveAvgPool3d(1)(x)
        x = x.view(-1, x.size(1))
        return x, [Tc1, Tc2, Tc3, Tc4]

    def _make_layer_fast(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.fast_inplanes != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.fast_inplanes,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, stride, stride),
                        bias=False))

        layers = []
        layers.append(block(self.fast_inplanes, planes, stride, downsample, head_conv=head_conv))
        self.fast_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.fast_inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)

    def _make_layer_slow(self, block, planes, blocks, shortcut_type, stride=1, head_conv=1):
        downsample = None
        if stride != 1 or self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2 != planes * block.expansion:
            if shortcut_type == 'A':
                downsample = partial(
                    downsample_basic_block,
                    planes=planes * block.expansion,
                    stride=stride)
            else:
                downsample = nn.Sequential(
                    nn.Conv3d(
                        self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2,
                        planes * block.expansion,
                        kernel_size=1,
                        stride=(1, stride, stride),
                        bias=False))

        layers = []
        layers.append(block(self.slow_inplanes + int(self.slow_inplanes * self.beta) * 2, planes, stride, downsample,
                            head_conv=head_conv))
        self.slow_inplanes = planes * block.expansion
        for i in range(1, blocks):
            layers.append(block(self.slow_inplanes, planes, head_conv=head_conv))

        return nn.Sequential(*layers)


def slowfast50(**kwargs):
    """Constructs a SlowFast-50 model.
    """
    model = SlowFast(Bottleneck, [3, 4, 6, 3], **kwargs)
    return model


# definition of xception
import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn as nn
import math


class SeparableConv2d(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=1, stride=1, padding=0, dilation=1, bias=False):
        super(SeparableConv2d, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, in_channels, kernel_size, stride, padding, dilation, groups=in_channels,
                               bias=bias)
        self.pointwise = nn.Conv2d(in_channels, out_channels, 1, 1, 0, 1, 1, bias=bias)

    def forward(self, x):
        x = self.conv1(x)
        x = self.pointwise(x)
        return x


class Block(nn.Module):
    def __init__(self, in_filters, out_filters, reps, strides=1, start_with_relu=True, grow_first=True):
        super(Block, self).__init__()
        if out_filters != in_filters or strides != 1:
            self.skip = nn.Conv2d(in_filters, out_filters, 1, stride=strides, bias=False)
            self.skipbn = nn.BatchNorm2d(out_filters)
        else:
            self.skip = None
        self.relu = nn.ReLU(inplace=True)
        rep = []
        filters = in_filters
        if grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
            filters = out_filters
        for i in range(reps - 1):
            rep.append(self.relu)
            rep.append(SeparableConv2d(filters, filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(filters))
        if not grow_first:
            rep.append(self.relu)
            rep.append(SeparableConv2d(in_filters, out_filters, 3, stride=1, padding=1, bias=False))
            rep.append(nn.BatchNorm2d(out_filters))
        if not start_with_relu:
            rep = rep[1:]
        else:
            rep[0] = nn.ReLU(inplace=False)
        if strides != 1:
            rep.append(nn.MaxPool2d(3, strides, 1))
        self.rep = nn.Sequential(*rep)

    def forward(self, inp):
        x = self.rep(inp)
        if self.skip is not None:
            skip = self.skip(inp)
            skip = self.skipbn(skip)
        else:
            skip = inp
        x += skip
        return x


class xception(nn.Module):
    """
    Xception optimized for the ImageNet dataset, as specified in
    https://arxiv.org/pdf/1610.02357.pdf
    """

    def __init__(self, num_classes=2, bypass_last_bn=False):
        """ Constructor
        Args:
            num_classes: number of classes
        """
        global bypass_bn_weight_list

        bypass_bn_weight_list = []
        self.inplanes = 64

        super(xception, self).__init__()
        self.num_classes = num_classes
        self.conv1 = nn.Conv2d(3, 32, 3, 2, 0, bias=False)
        self.bn1 = nn.BatchNorm2d(32)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(32, 64, 3, bias=False)
        self.bn2 = nn.BatchNorm2d(64)
        # do relu here
        self.block1 = Block(64, 128, 2, 2, start_with_relu=False, grow_first=True)
        self.block2 = Block(128, 256, 2, 2, start_with_relu=True, grow_first=True)
        self.block3 = Block(256, 728, 2, 2, start_with_relu=True, grow_first=True)
        self.block4 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block5 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block6 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block7 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block8 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block9 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block10 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block11 = Block(728, 728, 3, 1, start_with_relu=True, grow_first=True)
        self.block12 = Block(728, 1024, 2, 2, start_with_relu=True, grow_first=False)
        self.conv3 = SeparableConv2d(1024, 1536, 3, 1, 1)
        self.bn3 = nn.BatchNorm2d(1536)
        # do relu here
        self.conv4 = SeparableConv2d(1536, 2048, 3, 1, 1)
        self.bn4 = nn.BatchNorm2d(2048)
        self.fc = nn.Linear(2048, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
        if bypass_last_bn:
            for param in bypass_bn_weight_list:
                param.data.zero_()
            print('bypass {} bn.weight in BottleneckBlocks'.format(len(bypass_bn_weight_list)))

    def features(self, input):
        x = self.conv1(input)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.conv2(x)
        x = self.bn2(x)
        x = self.relu(x)
        x = self.block1(x)
        x = self.block2(x)
        x = self.block3(x)
        x = self.block4(x)
        x = self.block5(x)
        x = self.block6(x)
        x = self.block7(x)
        x = self.block8(x)
        x = self.block9(x)
        x = self.block10(x)
        x = self.block11(x)
        x = self.block12(x)
        x = self.conv3(x)
        x = self.bn3(x)
        x = self.relu(x)
        x = self.conv4(x)
        x = self.bn4(x)
        return x

    def logits(self, features):
        x = self.relu(features)
        x = F.adaptive_avg_pool2d(x, (1, 1))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

    def forward(self, input):
        x = self.features(input)
        x = self.logits(x)
        return x


# model inference
def load_state(path, model, cuda=True):
    def map_func(storage, location):
        return storage.cuda()

    if not cuda:
        map_func = torch.device('cpu')
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        have_load = set(pretrain(model, checkpoint['state_dict'], cuda))
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - have_load
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))
            pass
    else:
        print("=> no checkpoint found at '{}'".format(path))


def pretrain(model, state_dict, cuda):
    own_state = model.state_dict()
    have_load = []
    for name, param in state_dict.items():
        # remove "module." prefix
        name = name.replace(name.split('.')[0] + '.', '')
        if name in own_state:
            have_load.append(name)
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
    return have_load


def init_slow_fast_model(load_path, cuda):
    model = slowfast50(num_classes=2, alpha=4, beta=0.125, tau=16)
    load_state(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


# model inference
def load_state_xcp(path, model, cuda=True):
    def map_func(storage, location):
        return storage.cuda()

    if not cuda:
        map_func = torch.device('cpu')
    if os.path.isfile(path):
        print("=> loading checkpoint '{}'".format(path))
        checkpoint = torch.load(path, map_location=map_func)
        have_load = set(pretrain_xcp(model, checkpoint['state_dict'], cuda))
        own_keys = set(model.state_dict().keys())
        missing_keys = own_keys - have_load
        for k in missing_keys:
            print('caution: missing keys from checkpoint {}: {}'.format(path, k))
            pass
    else:
        print("=> no checkpoint found at '{}'".format(path))


def pretrain_xcp(model, state_dict, cuda):
    own_state = model.state_dict()
    have_load = []
    for name, param in state_dict.items():
        name = name.replace(name.split('.')[0] + '.' + name.split('.')[1] + '.', '')
        if name in own_state:
            have_load.append(name)
            if isinstance(param, torch.nn.Parameter):
                # backwards compatibility for serialized parameters
                param = param.data
            try:
                own_state[name].copy_(param)
            except:
                print('While copying the parameter named {}, '
                      'whose dimensions in the model are {} and '
                      'whose dimensions in the checkpoint are {}.'
                      .format(name, own_state[name].size(), param.size()))
                print("But don't worry about it. Continue pretraining.")
    return have_load


def init_xception_cls_model(load_path, cuda):
    model = xception()
    load_state_xcp(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


# definition of efficientnet
import re
import math
import collections
from functools import partial
import torch
from torch import nn
from torch.nn import functional as F
from torch.utils import model_zoo

########################################################################
############### HELPERS FUNCTIONS FOR MODEL ARCHITECTURE ###############
########################################################################


# Parameters for the entire model (stem, all blocks, and head)
GlobalParams = collections.namedtuple('GlobalParams', [
    'batch_norm_momentum', 'batch_norm_epsilon', 'dropout_rate',
    'num_classes', 'width_coefficient', 'depth_coefficient',
    'depth_divisor', 'min_depth', 'drop_connect_rate', 'image_size'])

# Parameters for an individual model block
BlockArgs = collections.namedtuple('BlockArgs', [
    'kernel_size', 'num_repeat', 'input_filters', 'output_filters',
    'expand_ratio', 'id_skip', 'stride', 'se_ratio'])

# Change namedtuple defaults
GlobalParams.__new__.__defaults__ = (None,) * len(GlobalParams._fields)
BlockArgs.__new__.__defaults__ = (None,) * len(BlockArgs._fields)


class SwishImplementation(torch.autograd.Function):
    @staticmethod
    def forward(ctx, i):
        result = i * torch.sigmoid(i)
        ctx.save_for_backward(i)
        return result

    @staticmethod
    def backward(ctx, grad_output):
        i = ctx.saved_variables[0]
        sigmoid_i = torch.sigmoid(i)
        return grad_output * (sigmoid_i * (1 + i * (1 - sigmoid_i)))


class MemoryEfficientSwish(nn.Module):
    def forward(self, x):
        return SwishImplementation.apply(x)


class Swish(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(x)


def round_filters(filters, global_params):
    """ Calculate and round number of filters based on depth multiplier. """
    multiplier = global_params.width_coefficient
    if not multiplier:
        return filters
    divisor = global_params.depth_divisor
    min_depth = global_params.min_depth
    filters *= multiplier
    min_depth = min_depth or divisor
    new_filters = max(min_depth, int(filters + divisor / 2) // divisor * divisor)
    if new_filters < 0.9 * filters:  # prevent rounding by more than 10%
        new_filters += divisor
    return int(new_filters)


def round_repeats(repeats, global_params):
    """ Round number of filters based on depth multiplier. """
    multiplier = global_params.depth_coefficient
    if not multiplier:
        return repeats
    return int(math.ceil(multiplier * repeats))


def drop_connect(inputs, p, training):
    """ Drop connect. """
    if not training: return inputs
    batch_size = inputs.shape[0]
    keep_prob = 1 - p
    random_tensor = keep_prob
    random_tensor += torch.rand([batch_size, 1, 1, 1], dtype=inputs.dtype, device=inputs.device)
    binary_tensor = torch.floor(random_tensor)
    output = inputs / keep_prob * binary_tensor
    return output


def get_same_padding_conv2d(image_size=None):
    """ Chooses static padding if you have specified an image size, and dynamic padding otherwise.
        Static padding is necessary for ONNX exporting of models. """
    if image_size is None:
        return Conv2dDynamicSamePadding
    else:
        return partial(Conv2dStaticSamePadding, image_size=image_size)


class Conv2dDynamicSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a dynamic image size """

    def __init__(self, in_channels, out_channels, kernel_size, stride=1, dilation=1, groups=1, bias=True):
        super().__init__(in_channels, out_channels, kernel_size, stride, 0, dilation, groups, bias)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

    def forward(self, x):
        ih, iw = x.size()[-2:]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            x = F.pad(x, [pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2])
        return F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)


class Conv2dStaticSamePadding(nn.Conv2d):
    """ 2D Convolutions like TensorFlow, for a fixed image size"""

    def __init__(self, in_channels, out_channels, kernel_size, image_size=None, **kwargs):
        super().__init__(in_channels, out_channels, kernel_size, **kwargs)
        self.stride = self.stride if len(self.stride) == 2 else [self.stride[0]] * 2

        # Calculate padding based on image size and save it
        assert image_size is not None
        ih, iw = image_size if type(image_size) == list else [image_size, image_size]
        kh, kw = self.weight.size()[-2:]
        sh, sw = self.stride
        oh, ow = math.ceil(ih / sh), math.ceil(iw / sw)
        pad_h = max((oh - 1) * self.stride[0] + (kh - 1) * self.dilation[0] + 1 - ih, 0)
        pad_w = max((ow - 1) * self.stride[1] + (kw - 1) * self.dilation[1] + 1 - iw, 0)
        if pad_h > 0 or pad_w > 0:
            self.static_padding = nn.ZeroPad2d((pad_w // 2, pad_w - pad_w // 2, pad_h // 2, pad_h - pad_h // 2))
        else:
            self.static_padding = Identity()

    def forward(self, x):
        x = self.static_padding(x)
        x = F.conv2d(x, self.weight, self.bias, self.stride, self.padding, self.dilation, self.groups)
        return x


class Identity(nn.Module):
    def __init__(self, ):
        super(Identity, self).__init__()

    def forward(self, input):
        return input


########################################################################
############## HELPERS FUNCTIONS FOR LOADING MODEL PARAMS ##############
########################################################################


def efficientnet_params(model_name):
    """ Map EfficientNet model name to parameter coefficients. """
    params_dict = {
        # Coefficients:   width,depth,res,dropout
        'efficientnet-b0': (1.0, 1.0, 224, 0.2),
        'efficientnet-b1': (1.0, 1.1, 240, 0.2),
        'efficientnet-b2': (1.1, 1.2, 260, 0.3),
        'efficientnet-b3': (1.2, 1.4, 300, 0.3),
        'efficientnet-b4': (1.4, 1.8, 380, 0.4),
        'efficientnet-b5': (1.6, 2.2, 456, 0.4),
        'efficientnet-b6': (1.8, 2.6, 528, 0.5),
        'efficientnet-b7': (2.0, 3.1, 600, 0.5),
    }
    return params_dict[model_name]


class BlockDecoder(object):
    """ Block Decoder for readability, straight from the official TensorFlow repository """

    @staticmethod
    def _decode_block_string(block_string):
        """ Gets a block through a string notation of arguments. """
        assert isinstance(block_string, str)

        ops = block_string.split('_')
        options = {}
        for op in ops:
            splits = re.split(r'(\d.*)', op)
            if len(splits) >= 2:
                key, value = splits[:2]
                options[key] = value

        # Check stride
        assert (('s' in options and len(options['s']) == 1) or
                (len(options['s']) == 2 and options['s'][0] == options['s'][1]))

        return BlockArgs(
            kernel_size=int(options['k']),
            num_repeat=int(options['r']),
            input_filters=int(options['i']),
            output_filters=int(options['o']),
            expand_ratio=int(options['e']),
            id_skip=('noskip' not in block_string),
            se_ratio=float(options['se']) if 'se' in options else None,
            stride=[int(options['s'][0])])

    @staticmethod
    def _encode_block_string(block):
        """Encodes a block to a string."""
        args = [
            'r%d' % block.num_repeat,
            'k%d' % block.kernel_size,
            's%d%d' % (block.strides[0], block.strides[1]),
            'e%s' % block.expand_ratio,
            'i%d' % block.input_filters,
            'o%d' % block.output_filters
        ]
        if 0 < block.se_ratio <= 1:
            args.append('se%s' % block.se_ratio)
        if block.id_skip is False:
            args.append('noskip')
        return '_'.join(args)

    @staticmethod
    def decode(string_list):
        """
        Decodes a list of string notations to specify blocks inside the network.

        :param string_list: a list of strings, each string is a notation of block
        :return: a list of BlockArgs namedtuples of block args
        """
        assert isinstance(string_list, list)
        blocks_args = []
        for block_string in string_list:
            blocks_args.append(BlockDecoder._decode_block_string(block_string))
        return blocks_args

    @staticmethod
    def encode(blocks_args):
        """
        Encodes a list of BlockArgs to a list of strings.

        :param blocks_args: a list of BlockArgs namedtuples of block args
        :return: a list of strings, each string is a notation of block
        """
        block_strings = []
        for block in blocks_args:
            block_strings.append(BlockDecoder._encode_block_string(block))
        return block_strings


def efficientnet(width_coefficient=None, depth_coefficient=None, dropout_rate=0.2,
                 drop_connect_rate=0.2, image_size=None, num_classes=1000):
    """ Creates a efficientnet model. """

    blocks_args = [
        'r1_k3_s11_e1_i32_o16_se0.25', 'r2_k3_s22_e6_i16_o24_se0.25',
        'r2_k5_s22_e6_i24_o40_se0.25', 'r3_k3_s22_e6_i40_o80_se0.25',
        'r3_k5_s11_e6_i80_o112_se0.25', 'r4_k5_s22_e6_i112_o192_se0.25',
        'r1_k3_s11_e6_i192_o320_se0.25',
    ]
    blocks_args = BlockDecoder.decode(blocks_args)

    global_params = GlobalParams(
        batch_norm_momentum=0.99,
        batch_norm_epsilon=1e-3,
        dropout_rate=dropout_rate,
        drop_connect_rate=drop_connect_rate,
        # data_format='channels_last',  # removed, this is always true in PyTorch
        num_classes=num_classes,
        width_coefficient=width_coefficient,
        depth_coefficient=depth_coefficient,
        depth_divisor=8,
        min_depth=None,
        image_size=image_size,
    )

    return blocks_args, global_params


def get_model_params(model_name, override_params):
    """ Get the block args and global params for a given model """
    if model_name.startswith('efficientnet'):
        w, d, s, p = efficientnet_params(model_name)
        # note: all models have drop connect rate = 0.2
        blocks_args, global_params = efficientnet(
            width_coefficient=w, depth_coefficient=d, dropout_rate=p, image_size=s)
    else:
        raise NotImplementedError('model name is not pre-defined: %s' % model_name)
    if override_params:
        # ValueError will be raised here if override_params has fields not included in global_params.
        global_params = global_params._replace(**override_params)
    return blocks_args, global_params


import torch.nn.functional as F
import torch.utils.model_zoo as model_zoo
from torch.nn import init
import torch.nn as nn
import math
import torch


class MBConvBlock(nn.Module):
    """
    Mobile Inverted Residual Bottleneck Block

    Args:
        block_args (namedtuple): BlockArgs, see above
        global_params (namedtuple): GlobalParam, see above

    Attributes:
        has_se (bool): Whether the block contains a Squeeze and Excitation layer.
    """

    def __init__(self, block_args, global_params):
        super().__init__()
        self._block_args = block_args
        self._bn_mom = 1 - global_params.batch_norm_momentum
        self._bn_eps = global_params.batch_norm_epsilon
        self.has_se = (self._block_args.se_ratio is not None) and (0 < self._block_args.se_ratio <= 1)
        self.id_skip = block_args.id_skip  # skip connection and drop connect

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Expansion phase
        inp = self._block_args.input_filters  # number of input channels
        oup = self._block_args.input_filters * self._block_args.expand_ratio  # number of output channels
        if self._block_args.expand_ratio != 1:
            self._expand_conv = Conv2d(in_channels=inp, out_channels=oup, kernel_size=1, bias=False)
            self._bn0 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Depthwise convolution phase
        k = self._block_args.kernel_size
        s = self._block_args.stride
        self._depthwise_conv = Conv2d(
            in_channels=oup, out_channels=oup, groups=oup,  # groups makes it depthwise
            kernel_size=k, stride=s, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=oup, momentum=self._bn_mom, eps=self._bn_eps)

        # Squeeze and Excitation layer, if desired
        if self.has_se:
            num_squeezed_channels = max(1, int(self._block_args.input_filters * self._block_args.se_ratio))
            self._se_reduce = Conv2d(in_channels=oup, out_channels=num_squeezed_channels, kernel_size=1)
            self._se_expand = Conv2d(in_channels=num_squeezed_channels, out_channels=oup, kernel_size=1)

        # Output phase
        final_oup = self._block_args.output_filters
        self._project_conv = Conv2d(in_channels=oup, out_channels=final_oup, kernel_size=1, bias=False)
        self._bn2 = nn.BatchNorm2d(num_features=final_oup, momentum=self._bn_mom, eps=self._bn_eps)
        self._swish = MemoryEfficientSwish()

    def forward(self, inputs, drop_connect_rate=None):
        """
        :param inputs: input tensor
        :param drop_connect_rate: drop connect rate (float, between 0 and 1)
        :return: output of block
        """

        # Expansion and Depthwise Convolution
        x = inputs
        if self._block_args.expand_ratio != 1:
            x = self._swish(self._bn0(self._expand_conv(inputs)))
        x = self._swish(self._bn1(self._depthwise_conv(x)))

        # Squeeze and Excitation
        if self.has_se:
            x_squeezed = F.adaptive_avg_pool2d(x, 1)
            x_squeezed = self._se_expand(self._swish(self._se_reduce(x_squeezed)))
            x = torch.sigmoid(x_squeezed) * x

        x = self._bn2(self._project_conv(x))

        # Skip connection and drop connect
        input_filters, output_filters = self._block_args.input_filters, self._block_args.output_filters
        if self.id_skip and self._block_args.stride == 1 and input_filters == output_filters:
            if drop_connect_rate:
                x = drop_connect(x, p=drop_connect_rate, training=self.training)
            x = x + inputs  # skip connection
        return x

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()


class EfficientNet(nn.Module):
    """
    An EfficientNet model. Most easily loaded with the .from_name or .from_pretrained methods

    Args:
        blocks_args (list): A list of BlockArgs to construct blocks
        global_params (namedtuple): A set of GlobalParams shared between blocks

    Example:
        model = EfficientNet.from_pretrained('efficientnet-b0')

    """

    def __init__(self, blocks_args=None, global_params=None, bn_group_size=1):
        super().__init__()

        assert isinstance(blocks_args, list), 'blocks_args should be a list'
        assert len(blocks_args) > 0, 'block args must be greater than 0'
        self._global_params = global_params
        self._blocks_args = blocks_args

        # Get static or dynamic convolution depending on image size
        Conv2d = get_same_padding_conv2d(image_size=global_params.image_size)

        # Batch norm parameters
        bn_mom = 1 - self._global_params.batch_norm_momentum
        bn_eps = self._global_params.batch_norm_epsilon

        # Stem
        in_channels = 3  # rgb
        out_channels = round_filters(32, self._global_params)  # number of output channels
        self._conv_stem = Conv2d(in_channels, out_channels, kernel_size=3, stride=2, bias=False)
        self._bn0 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Build blocks
        self._blocks = nn.ModuleList([])
        for block_args in self._blocks_args:

            # Update block input and output filters based on depth multiplier.
            block_args = block_args._replace(
                input_filters=round_filters(block_args.input_filters, self._global_params),
                output_filters=round_filters(block_args.output_filters, self._global_params),
                num_repeat=round_repeats(block_args.num_repeat, self._global_params)
            )

            # The first block needs to take care of stride and filter size increase.
            self._blocks.append(MBConvBlock(block_args, self._global_params))
            if block_args.num_repeat > 1:
                block_args = block_args._replace(input_filters=block_args.output_filters, stride=1)
            for _ in range(block_args.num_repeat - 1):
                self._blocks.append(MBConvBlock(block_args, self._global_params))

        # Head
        in_channels = block_args.output_filters  # output of final block
        out_channels = round_filters(1280, self._global_params)
        self._conv_head = Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self._bn1 = nn.BatchNorm2d(num_features=out_channels, momentum=bn_mom, eps=bn_eps)

        # Final linear layer
        self._avg_pooling = nn.AdaptiveAvgPool2d(1)
        self._dropout = nn.Dropout(self._global_params.dropout_rate)
        self._fc = nn.Linear(out_channels, self._global_params.num_classes)
        self._swish = MemoryEfficientSwish()

    def set_swish(self, memory_efficient=True):
        """Sets swish function as memory efficient (for training) or standard (for export)"""
        self._swish = MemoryEfficientSwish() if memory_efficient else Swish()
        for block in self._blocks:
            block.set_swish(memory_efficient)

    def extract_features(self, inputs):
        """ Returns output of the final convolution layer """

        # Stem
        x = self._swish(self._bn0(self._conv_stem(inputs)))

        # Blocks
        for idx, block in enumerate(self._blocks):
            drop_connect_rate = self._global_params.drop_connect_rate
            if drop_connect_rate:
                drop_connect_rate *= float(idx) / len(self._blocks)
            x = block(x, drop_connect_rate=drop_connect_rate)

        # Head
        x = self._swish(self._bn1(self._conv_head(x)))

        return x

    def forward(self, inputs):
        """ Calls extract_features to extract features, applies final linear layer, and returns logits. """
        bs = inputs.size(0)
        # Convolution layers
        x = self.extract_features(inputs)

        # Pooling and final linear layer
        x = self._avg_pooling(x)
        x = x.view(bs, -1)
        x = self._dropout(x)
        x = self._fc(x)
        return x

    @classmethod
    def from_name(cls, model_name, override_params=None):
        cls._check_model_name_is_valid(model_name)
        blocks_args, global_params = get_model_params(model_name, override_params)
        return cls(blocks_args, global_params)

    @classmethod
    def from_pretrained(cls, model_name, num_classes=2):
        model = cls.from_name(model_name, override_params={'num_classes': num_classes})
        return model

    @classmethod
    def get_image_size(cls, model_name):
        cls._check_model_name_is_valid(model_name)
        _, _, res, _ = efficientnet_params(model_name)
        return res

    @classmethod
    def _check_model_name_is_valid(cls, model_name, also_need_pretrained_weights=False):
        """ Validates model name. None that pretrained weights are only available for
        the first four models (efficientnet-b{i} for i in 0,1,2,3) at the moment. """
        num_models = 4 if also_need_pretrained_weights else 8
        valid_models = ['efficientnet-b' + str(i) for i in range(num_models)]
        if model_name not in valid_models:
            raise ValueError('model_name should be one of: ' + ', '.join(valid_models))


def init_b3_cls_model(load_path, cuda):
    model = EfficientNet.from_pretrained('efficientnet-b3', num_classes=2)
    load_state(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


def init_b1_cls_model(load_path, cuda):
    model = EfficientNet.from_pretrained('efficientnet-b1', num_classes=2)
    load_state(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


def init_b0_cls_model(load_path, cuda):
    model = EfficientNet.from_pretrained('efficientnet-b0', num_classes=2)
    load_state(load_path, model, cuda)
    if cuda:
        model.cuda()
    model.eval()
    return model


# definition of resnet
import torchvision.models.resnet as resnet


def init_res34_cls_model(load_path, cuda):
    model = resnet.resnet34(num_classes=2)
    load_state(load_path, model, cuda)
    if cuda: model.cuda()
    model.eval()
    return model


def get_image_score(face_scale, cls_model, softmax_func, aligned_faces, isRGB, mean, std, isScale, isFlip=False):
    try:
        # aligned_faces #[faces,frames,H,W,C] BGR
        img_aligned_faces = aligned_faces.clone().detach()
        img_aligned_faces = img_aligned_faces.permute([0, 1, 4, 2, 3])  # [faces,frames,c,h,w] BGR
        if isRGB: img_aligned_faces = img_aligned_faces[:, :, [2, 1, 0], :, :]
        img_frames = 35
        interval = max(1, math.ceil(img_aligned_faces.shape[1] / img_frames))
        img_aligned_faces = img_aligned_faces[:, 0::interval, :, :, :]
        img_frames = (img_aligned_faces.shape[1] // 5) * 5
        img_aligned_faces = img_aligned_faces[:, :img_frames, :, :, :]

        all_score, score = [], 0
        for face_idx in range(img_aligned_faces.shape[0]):
            one_face_aligned = img_aligned_faces[face_idx, :, :, :, :]
            one_face_aligned_mean = (one_face_aligned - mean) / std  # [frames,c,h,w]
            if isFlip:
                one_face_aligned_mean_flip = torch.flip(one_face_aligned_mean, dims=[3])
                one_face_aligned_input = torch.cat((one_face_aligned_mean, one_face_aligned_mean_flip), dim=0)
                output = cls_model(one_face_aligned_input)
                output = (output[:img_frames, :] + output[img_frames:, :]) / 2
            else:
                output = cls_model(one_face_aligned_mean)
            output = output.view(-1, 5, 2)
            output = output.mean(1)
            output = softmax_func(output)
            output = output[:, 1].cpu().numpy()  # [6,1]
            if output[output > 0.85].shape[0] / output.shape[0] > 0.7:
                score = output[output > 0.85].mean()
            elif output[output < 0.15].shape[0] / output.shape[0] > 0.7:
                score = output[output < 0.15].mean()
            else:
                score = output.mean()
            all_score.append(score)
        all_score = np.array(all_score)
        score_max, score_min, score_avg = np.max(all_score), np.min(all_score), np.mean(all_score)
        if score_max > 0.9:
            score = score_max
        elif len(np.where(all_score > 0.6)[0]) == all_score.shape[0]:
            score = score_max
        elif len(np.where(all_score < 0.4)[0]) == all_score.shape[0]:
            score = score_min
        else:
            score = score_avg
        if isScale:
            if score >= 0.98 or score <= 0.02:
                score = (score - 0.5) * 0.96 + 0.5
    except Exception as e:
        print(e)
        score = -1
    return score


def get_sf_score(face_scale, cls_model, softmax_func, aligned_faces, isRGB, mean, std):
    try:
        # aligned_faces [faces,frames,H,W,C]  BGR
        sf_aligned_faces = aligned_faces.clone().detach()
        sf_aligned_faces = sf_aligned_faces.permute([0, 4, 1, 2, 3])  # [faces,c,frames,h,w]
        if isRGB: sf_aligned_faces = sf_aligned_faces[:, [2, 1, 0], :, :, :]
        sf_aligned_faces = (sf_aligned_faces - mean) / std
        sf_output = cls_model(sf_aligned_faces)
        sf_output = softmax_func(sf_output)
        sf_output = sf_output[:, 1].cpu().numpy()
        sf_max, sf_min, sf_avg = np.max(sf_output), np.min(sf_output), np.mean(sf_output)
        if sf_max > 0.9:
            sf_score = sf_max
        elif len(np.where(sf_output > 0.6)[0]) == sf_output.shape[0]:
            sf_score = sf_max
        elif len(np.where(sf_output < 0.4)[0]) == sf_output.shape[0]:
            sf_score = sf_min
        else:
            sf_score = sf_avg
    except Exception as e:
        print(e)
        sf_score = -1
    return sf_score


def get_final_score(score_list, weight_list):
    final_score = 0
    assert len(score_list) == len(weight_list)
    new_score_list, new_weight_list = [], []
    for idx, score in enumerate(score_list):
        if score != -1:
            new_score_list.append(score)
            new_weight_list.append(weight_list[idx])
    new_scores, new_weights = np.array(new_score_list), np.array(new_weight_list)
    if len(new_weights) == 0: return -1
    print('new_scores:', new_scores, 'new_weights', new_weights / np.sum(new_weights))
    final_score = np.sum(new_scores * (new_weights / np.sum(new_weights)))
    return final_score


def get_final_score_policy(score_list, weight_list, img_start_idx, sf_weight):
    assert len(score_list) == len(weight_list)
    sf_score_list, sf_weight_list = score_list[:img_start_idx], weight_list[:img_start_idx]
    img_score_list, img_weight_list = score_list[img_start_idx:], weight_list[img_start_idx:]
    new_sf_score_list, new_sf_weight_list, new_img_score_list, new_img_weight_list = [], [], [], []
    for idx, score in enumerate(sf_score_list):
        if score != -1:
            new_sf_score_list.append(score)
            new_sf_weight_list.append(sf_weight_list[idx])

    for idx, score in enumerate(img_score_list):
        if score != -1:
            new_img_score_list.append(score)
            new_img_weight_list.append(img_weight_list[idx])
    new_sf_scores, new_sf_weights = np.array(new_sf_score_list), np.array(new_sf_weight_list)
    new_img_scores, new_img_weights = np.array(new_img_score_list), np.array(new_img_weight_list)

    sf_success, img_success = True, True
    # sf
    if new_sf_scores.shape[0] != 0:
        if len(np.where(new_sf_scores > 0.8)[0]) / new_sf_scores.shape[0] > 0.7:
            new_sf_y_scores, new_sf_y_weights = new_sf_scores[new_sf_scores > 0.8], new_sf_weights[new_sf_scores > 0.8]
            sf_score = np.sum(new_sf_y_scores * (new_sf_y_weights / np.sum(new_sf_y_weights)))
        elif len(np.where(new_sf_scores < 0.2)[0]) / new_sf_scores.shape[0] > 0.7:
            new_sf_y_scores, new_sf_y_weights = new_sf_scores[new_sf_scores < 0.2], new_sf_weights[new_sf_scores < 0.2]
            sf_score = np.sum(new_sf_y_scores * (new_sf_y_weights / np.sum(new_sf_y_weights)))
        else:
            sf_score = np.sum(new_sf_scores * (new_sf_weights / np.sum(new_sf_weights)))
    else:
        sf_success = False

    # img
    if new_img_scores.shape[0] != 0:
        if len(np.where(new_img_scores > 0.8)[0]) / new_img_scores.shape[0] > 0.7:
            new_img_y_scores, new_img_y_weights = new_img_scores[new_img_scores > 0.8], new_img_weights[
                new_img_scores > 0.8]
            img_score = np.sum(new_img_y_scores * (new_img_y_weights / np.sum(new_img_y_weights)))
        elif len(np.where(new_img_scores < 0.2)[0]) / new_img_scores.shape[0] > 0.7:
            new_img_y_scores, new_img_y_weights = new_img_scores[new_img_scores < 0.2], new_img_weights[
                new_img_scores < 0.2]
            img_score = np.sum(new_img_y_scores * (new_img_y_weights / np.sum(new_img_y_weights)))
        else:
            img_score = np.sum(new_img_scores * (new_img_weights / np.sum(new_img_weights)))
    else:
        img_success = False

    if sf_success and img_success:
        final_score = sf_score * sf_weight + (1 - sf_weight) * img_score
    elif sf_success and not img_success:
        final_score = sf_score
    elif img_success and not sf_success:
        final_score = img_score
    else:
        final_score = -1
    return final_score


def predict_batch(img_list, sf_model1, sf_model2, sf_model3, xcp_model, b3_model, res34_model, b1_model, b1long_model,
                  b1short_model, b0_model, sf_model4, softmax_func, detect_record):
    # face det
    aligned_faceses, noface_flag = detect_video_face(img_list, detect_record)
    if noface_flag == -1: return -1
    sf1_score, sf2_score, sf3_score, sf4_score, xcp_score, b3_score, res34_score, b1_score, b1long_score, b1short_score, b0_score = 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0
    scale_num = len(aligned_faceses)
    # slowfast model infer
    sf_mean = torch.from_numpy(
        np.array([110.63666788 / 255, 103.16065604 / 255, 96.29023126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).cuda()
    sf_std = torch.from_numpy(
        np.array([38.7568578 / 255, 37.88248729 / 255, 40.02898126 / 255], dtype=np.float32)).reshape(
        [1, -1, 1, 1, 1]).cuda()
    xcp_mean = torch.from_numpy(np.array([0.5, 0.5, 0.5], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    xcp_std = torch.from_numpy(np.array([0.5, 0.5, 0.5], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    b3_mean = torch.from_numpy(np.array([0.485, 0.456, 0.406], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()
    b3_std = torch.from_numpy(np.array([0.229, 0.224, 0.225], dtype=np.float32)).reshape([1, -1, 1, 1]).cuda()

    for aligned_faces in aligned_faceses:
        print('aligned_faces shape:', aligned_faces.shape)
        aligned_faces = np.float32(aligned_faces)
        face_scale, face_num, align_frames = aligned_faces.shape[3], aligned_faces.shape[0], aligned_faces.shape[1]
        # init scale tensor
        aligned_faces = torch.from_numpy(aligned_faces)
        if pipeline_cfg.cuda: aligned_faces = aligned_faces.cuda()  # [faces,frames,H,W,C]
        aligned_faces /= 255

        # xcp inference
        if face_scale == 299:
            xcp_score = get_image_score(face_scale, xcp_model, softmax_func, aligned_faces, False, xcp_mean, xcp_std,
                                        False, True)
            b3_score = get_image_score(face_scale, b3_model, softmax_func, aligned_faces, True, b3_mean, b3_std, True,
                                       False)
            b1_score = get_image_score(face_scale, b1_model, softmax_func, aligned_faces, True, b3_mean, b3_std, False,
                                       True)
            b1long_score = get_image_score(face_scale, b1long_model, softmax_func, aligned_faces, True, b3_mean, b3_std,
                                           False, False)
            b1short_score = get_image_score(face_scale, b1short_model, softmax_func, aligned_faces, True, b3_mean,
                                            b3_std, False, False)
            b0_score = get_image_score(face_scale, b0_model, softmax_func, aligned_faces, True, b3_mean, b3_std, False,
                                       True)

        if face_scale == 256:
            res34_score = get_image_score(face_scale, res34_model, softmax_func, aligned_faces, True, b3_mean, b3_std,
                                          True, True)
            sf1_score = get_sf_score(face_scale, sf_model1, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf2_score = get_sf_score(face_scale, sf_model2, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf3_score = get_sf_score(face_scale, sf_model3, softmax_func, aligned_faces, True, sf_mean, sf_std)
            sf4_score = get_sf_score(face_scale, sf_model4, softmax_func, aligned_faces, True, sf_mean, sf_std)

    score_list = [sf1_score, sf2_score, sf3_score, sf4_score, xcp_score, b3_score, res34_score, b1_score, b1long_score,
                  b1short_score, b0_score]
    print(score_list)
    sf_weight_np, img_weight_np = np.array([10, 8, 4, 8]), np.array([10, 6, 4, 10, 8, 8, 7])
    sf_weight_np = sf_weight_np / np.sum(sf_weight_np) * 0.4
    img_weight_np = img_weight_np / np.sum(img_weight_np) * 0.6
    weight_np = np.concatenate((sf_weight_np, img_weight_np))
    weight_list = list(weight_np)
    print(weight_list)
    final_score = get_final_score_policy(score_list, weight_list, len(sf_weight_np), 0.4)
    return final_score


def detect_video_face(img_list, detect_record):
    num_frames = len(img_list)
    num_faces = 0
    face_count = {}
    img_h, img_w = img_list[0].shape[0], img_list[0].shape[1]
    face_list = []

    detect_tmp_begin = time.time()
    dets_list = detect_face(img_list, detect_record)
    detect_tmp_end = time.time()
    detect_face_time = detect_tmp_end - detect_tmp_begin
    global DETECT_FACE_TIME
    DETECT_FACE_TIME += detect_face_time
    print('detect face time:', detect_face_time)

    align_begin = time.time()
    for idx, img_raw in enumerate(img_list):
        # preserve only faces with confidence above threshold
        dets = dets_list[idx][np.where(dets_list[idx][:, 4] >= pipeline_cfg.vis_thres)][:, :4].astype(
            np.int64)  # [m,15]
        face_list.append(dets)
        if len(dets) not in face_count:
            face_count[len(dets)] = 0
        face_count[len(dets)] += 1

    face_align_begin = time.time()

    # vote for the number of faces that most frames agree on
    max_count = 0
    for num in face_count:
        if face_count[num] > max_count:
            num_faces = num
            max_count = face_count[num]
    if num_faces <= 0:
        return None, -1

    active_faces = None
    face_tubes = []
    for frame_idx in range(num_frames):
        cur_faces = face_list[frame_idx]  #
        if len(cur_faces) <= 0:
            continue

        if active_faces is not None:
            ious = vanilla_bbox_iou_overlaps(cur_faces, active_faces)
            max_iou, max_idx = np.max(ious, axis=1), np.argmax(ious, axis=1)
            mark = [False for _ in range(len(active_faces))]
        else:
            max_iou, max_idx = None, None

        for face_idx in range(len(cur_faces)):
            # IoU threshold 0.5 for determining whether is the same person
            if max_iou is None or max_iou[face_idx] < 0.5:
                face = copy.deepcopy(cur_faces[face_idx])
                if active_faces is None:
                    active_faces = face[np.newaxis, :]
                else:
                    active_faces = np.concatenate((active_faces, face[np.newaxis, :]), axis=0)
                face_tubes.append([[frame_idx, face_idx]])
            else:
                correspond_idx = max_idx[face_idx]
                # Each face tube can only add at most one face from a frame
                if mark[correspond_idx]:
                    continue
                mark[correspond_idx] = True
                active_faces[correspond_idx] = cur_faces[face_idx]
                face_tubes[correspond_idx].append([frame_idx, face_idx])
    # Choose num_faces longest face_tubes as chosen faces
    face_tubes.sort(key=lambda tube: len(tube), reverse=True)
    if len(face_tubes) < num_faces:
        num_faces = len(face_tubes)
    num_faces = min(num_faces, 2)
    face_tubes = face_tubes[:num_faces]

    aligned_faces_img_256, aligned_faces_img_299, aligned_faces_img_320 = [], [], []
    for face_idx in range(num_faces):
        cur_face_list, source_frame_list = [], []
        # record max crop_bbox size
        tube_idx, max_size = 0, 0
        for frame_idx in range(num_frames):
            cur_face = face_tubes[face_idx][tube_idx]
            next_face = None if tube_idx == len(face_tubes[face_idx]) - 1 else face_tubes[face_idx][tube_idx + 1]
            # find nearest frame inside face tube
            if next_face is not None and abs(cur_face[0] - frame_idx) > abs(next_face[0] - frame_idx):
                tube_idx += 1
                cur_face = next_face
            face = copy.deepcopy(face_list[cur_face[0]][cur_face[1]])
            cur_face_list.append(face)
            source_frame_list.append(cur_face[0])

            _, _, size = get_boundingbox(face, img_w, img_h)
            if size > max_size:
                max_size = size

        # align face size
        max_size = max_size // 2 * 2
        max_size = min(max_size, img_w, img_h)

        # adjust to max face size and crop faces
        cur_faces_img_256, cur_faces_img_299, cur_faces_img_320 = [], [], []
        for frame_idx in range(num_frames):
            x1, y1, size = adjust_boundingbox(cur_face_list[frame_idx], img_w, img_h, max_size)
            img = img_list[source_frame_list[frame_idx]][y1:y1 + size, x1:x1 + size, :]
            img_256 = cv2.resize(img, (256, 256), interpolation=cv2.INTER_LINEAR)
            cur_faces_img_256.append(img_256)
            img_299 = cv2.resize(img, (299, 299), interpolation=cv2.INTER_LINEAR)
            cur_faces_img_299.append(img_299)


        cur_faces_numpy_256 = np.stack(cur_faces_img_256, axis=0)  # [num_frames, h, w, c]
        cur_faces_numpy_299 = np.stack(cur_faces_img_299, axis=0)  # [num_frames, h, w, c]

        aligned_faces_img_256.append(cur_faces_numpy_256)
        aligned_faces_img_299.append(cur_faces_numpy_299)

    aligned_faces_numpy_256 = np.stack(aligned_faces_img_256, axis=0)  # [num_faces, num_frames, h, w, c]
    aligned_faces_numpy_299 = np.stack(aligned_faces_img_299, axis=0)  # [num_faces, num_frames, h, w, c]
    align_end = time.time()
    print('align_time:256 299', align_end - align_begin)
    return [aligned_faces_numpy_256, aligned_faces_numpy_299], 1


# test video
class Test:
    def __init__(self, test_video_dir: str, cls_model_ckpt: str, xcp_model_ckpt: str, slow_fast_2_ckpt: str,
                 slow_fast_3_ckpt: str, b3_model_ckpt: str, res34_model_ckpt: str, b1_model_ckpt: str,
                 b1long_model_ckpt: str, b1short_model_ckpt: str, b0_model_ckpt: str, slow_fast_4_ckpt: str,
                 frame_nums: int, cuda=True):
        self.test_video_dir = test_video_dir
        self.cls_model_ckpt = cls_model_ckpt
        self.xcp_model_ckpt = xcp_model_ckpt
        self.cls_model2_ckpt = slow_fast_2_ckpt
        self.cls_model3_ckpt = slow_fast_3_ckpt
        self.cls_model4_ckpt = slow_fast_4_ckpt
        self.b3_model_ckpt = b3_model_ckpt
        self.res34_model_ckpt = res34_model_ckpt
        self.b1_model_ckpt = b1_model_ckpt
        self.b1long_model_ckpt = b1long_model_ckpt
        self.b1short_model_ckpt = b1short_model_ckpt
        self.b0_model_ckpt = b0_model_ckpt

        self.frame_nums = frame_nums
        self.cuda = cuda
        self.detect_record = {}

    def init_model(self):
        self.face_det_model = init_face_detecor()
        self.face_cls_model = init_slow_fast_model(self.cls_model_ckpt, self.cuda)
        self.face_cls_model2 = init_slow_fast_model(self.cls_model2_ckpt, self.cuda)
        self.face_cls_model3 = init_slow_fast_model(self.cls_model3_ckpt, self.cuda)
        self.face_cls_model4 = init_slow_fast_model(self.cls_model4_ckpt, self.cuda)
        self.xcp_cls_model = init_xception_cls_model(self.xcp_model_ckpt, self.cuda)
        self.b3_cls_model = init_b3_cls_model(self.b3_model_ckpt, self.cuda)
        self.res34_cls_model = init_res34_cls_model(self.res34_model_ckpt, self.cuda)
        self.b1_cls_model = init_b1_cls_model(self.b1_model_ckpt, self.cuda)
        self.b1long_cls_model = init_b1_cls_model(self.b1long_model_ckpt, self.cuda)
        self.b1short_cls_model = init_b1_cls_model(self.b1short_model_ckpt, self.cuda)
        self.b0_cls_model = init_b0_cls_model(self.b0_model_ckpt, self.cuda)

    def test_kernel_video(self):
        post_func = nn.Softmax(dim=1)
        init_begin = time.time()
        self.init_model()
        init_end = time.time()
        print('init model time:', init_end - init_begin)
        submission = pd.read_csv("./sample_submission.csv", dtype='unicode')
        for i in range(len(submission)):
            score = 0.5
            try:
                video_begin = time.time()
                video_name = submission['filename'][i]
                print(video_name)
                video_pth = os.path.join(self.test_video_dir, video_name)
                print(video_pth)
                if video_pth.split('.')[-1] != 'mp4': continue
                # extract image
                reader = cv2.VideoCapture(video_pth)
                video_cnt = reader.get(cv2.CAP_PROP_FRAME_COUNT)
                interval = max(1, math.ceil(video_cnt / self.frame_nums))
                print('video_cnt:', video_cnt, 'interval:', interval)
                count, test_count = 0, 0
                success = True
                img_list = []
                cv2_read_begin = time.time()
                while success:
                    if count % interval == 0:
                        success, image = reader.read()
                        if success:
                            img_list.append(image)
                    else:
                        success = reader.grab()
                    count += 1
                reader.release()
                cv2_read_end = time.time()
                print('cv2_read_time:', cv2_read_end - cv2_read_begin)
                global VIDEO_LOAD_TIME
                VIDEO_LOAD_TIME += cv2_read_end - cv2_read_begin
                score = predict_batch(img_list, self.face_cls_model, self.face_cls_model2, self.face_cls_model3,
                                      self.xcp_cls_model, self.b3_cls_model, self.res34_cls_model, self.b1_cls_model,
                                      self.b1long_cls_model, self.b1short_cls_model, self.b0_cls_model,
                                      self.face_cls_model4, post_func, self.detect_record)
            except Exception as e:
                print(e)
                score = -1
            print('score:',score)
            if score < 0 or score > 1: score = 0.5
            submission['label'][i] = score
            video_end = time.time()
            print('video_time:', video_end - video_begin)
            global VIDEO_TIME
            VIDEO_TIME += video_end - video_begin
            print('video_score:', score)
            print('---------------------------------------')
            end_time = time.time()
        submission.to_csv('submission.csv', index=False)
        return submission


# test
if __name__ == '__main__':
    begin_time = time.time()
    # can be replaced by any test video root, and the list of videos should be in ./sample_submission.csv
    test_dir = './test_videos'
    load_slowfast_path = './submit_models/sf_bc_jc_44000.pth.tar'
    load_slowfast_path2 = './submit_models/sf_32000.pth.tar'
    load_slowfast_path3 = './submit_models/sf_16x8_bc_jc_44000.pth.tar'
    load_slowfast_path4 = './submit_models/sf_trainval_52000.pth.tar'
    load_xcp_path = './submit_models/xcep_bgr_58000.pth.tar'
    load_b3_path = './submit_models/b3_rgb_50000.pth.tar'
    load_res34_path = './submit_models/res34_rgb_23000.pth.tar'
    load_b1_path = './submit_models/b1_rgb_58000.pth.tar'
    load_b1long_path = './submit_models/b1_rgb_long_alldata_66000.pth.tar'
    load_b1short_path = './submit_models/b1_rgb_alldata_58000.pth.tar'
    load_b0_path = './submit_models/b0_rgb_58000.pth.tar'

    frame_nums = 160
    submit = Test(test_dir, load_slowfast_path, load_xcp_path, load_slowfast_path2, load_slowfast_path3, load_b3_path,
                  load_res34_path, load_b1_path,
                  load_b1long_path, load_b1short_path, load_b0_path, load_slowfast_path4, frame_nums,
                  cuda=pipeline_cfg.cuda)
    final_ret = submit.test_kernel_video()
    end_time = time.time()
    test_num = len(final_ret)
    print('all_test_time of %d videos:'%test_num, end_time - begin_time)
    print('cv2 read:{},detect face:{},video:{}'
          .format(VIDEO_LOAD_TIME / test_num, DETECT_FACE_TIME / test_num, VIDEO_TIME / test_num))
