import multiprocessing as mp
mp.set_start_method('spawn', force=True)
import argparse
import os
import time
import yaml
import pickle
import numpy
import logging
from easydict import EasyDict
from datetime import datetime
import pprint
from tensorboardX import SummaryWriter
import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import torch.optim
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
import albumentations
from albumentations.augmentations.transforms import ShiftScaleRotate, HorizontalFlip, Normalize, RandomBrightnessContrast, MotionBlur, Blur, GaussNoise, JpegCompression, Resize, RandomBrightness, RandomResizedCrop
from albumentations.pytorch.transforms import ToTensorV2
import math
import torch.distributed as dist

from models import model_entry
from scheduler import get_scheduler
from dataset import FaceDataset
from utils import create_logger, AverageMeter, accuracy, save_checkpoint, load_state, DistributedGivenIterationSampler, DistributedSampler, parameters_string
from distributed_utils import dist_init

from optim import optim_entry


def mkdir(path):
    try:
        os.makedirs(path)
    except:
        pass

parser = argparse.ArgumentParser(description='PyTorch DFDC Training')
parser.add_argument('--config', default='cfgs/config_res50.yaml')
parser.add_argument('--distributed_path', default='/')
parser.add_argument('--datetime', default='19700101-000000')
parser.add_argument('--no_val', action='store_true')
parser.add_argument('--load_path', default='', type=str)
parser.add_argument('--recover', action='store_true')
parser.add_argument('-e', '--evaluate', action='store_true')
parser.add_argument('--evaluate_path', default='', type=str)
parser.add_argument('--range_list', default='0_200000_1000', type=str, help='checkpoint list, format: begin_end_gap')
parser.add_argument('--run_tag', default='', type=str)

def get_all_checkpoint(path, range_list, rank):
    begin, end, gap = range_list.split('_')
    exist_ckpts = {}
    res = []
    for root, directories, filenames in os.walk(path):
        for filename in filenames:
            if filename.find('ckpt_iter_') != -1:
                tname = os.path.join(root, filename)
                step = filename.replace('ckpt_iter_', '')
                step = step.replace('.pth.tar', '')
                step = int(step)
                exist_ckpts[step] = tname
    for i in range(int(begin), int(end), int(gap)):
        if i in exist_ckpts.keys():
            res.append(exist_ckpts[i])
        else:
            if rank == 0:
                print('No ckpt_iter_' + str(i) + '.pth.tar, skipped.')
    return res

def main():
    global args, config, best_loss
    args = parser.parse_args()

    with open(args.config) as f:
        config = yaml.load(f)
    
    for k, v in config['common'].items():
        setattr(args, k, v)
    config = EasyDict(config['common'])
    
    rank, world_size, device_id = dist_init(os.path.join(args.distributed_path, config.distributed_file))
    
    args.save_path_dated = args.save_path + '/' + args.datetime
    if args.run_tag != '':
        args.save_path_dated += '-' + args.run_tag

    # create model
    model = model_entry(config.model)
    model.cuda()

    model = nn.parallel.DistributedDataParallel(model, device_ids=[device_id])


    # create optimizer
    opt_config = config.optimizer
    opt_config.kwargs.lr = config.lr_scheduler.base_lr
    opt_config.kwargs.params = model.parameters()

    optimizer = optim_entry(opt_config)
    
    # optionally resume from a checkpoint
    last_iter = -1
    best_loss = 1e9
    if args.load_path:
        if args.recover:
            best_loss, last_iter = load_state(args.load_path, model, optimizer=optimizer)
        else:
            load_state(args.load_path, model)

    cudnn.benchmark = True
    
    # train augmentation
    if config.augmentation.get('imgnet_mean', False):
        model_mean = (0.485, 0.456, 0.406)
        model_std = (0.229, 0.224, 0.225)
    else:
        model_mean = (0.5, 0.5, 0.5)
        model_std = (0.5, 0.5, 0.5)
    trans = albumentations.Compose([
        RandomResizedCrop(config.augmentation.input_size, config.augmentation.input_size,
                          scale=(config.augmentation.min_scale ** 2., 1.), ratio=(1., 1.)),
        HorizontalFlip(p=0.5),
        RandomBrightnessContrast(brightness_limit=0.25, contrast_limit=0.1, p=0.5),
        JpegCompression(p=.2, quality_lower=50),
        MotionBlur(p=0.5),
        Normalize(mean=model_mean, std=model_std),
        ToTensorV2()
    ])

    train_dataset = FaceDataset(
        config.train_root,
        config.train_source,
        transform=trans,
        resize=config.augmentation.input_size,
        image_format=config.get('image_format', None),
        random_frame=config.get('train_random_frame', False),
        bgr=config.augmentation.get('bgr', False)
    )

    train_sampler = DistributedGivenIterationSampler(
        train_dataset, config.lr_scheduler.max_iter,
        config.batch_size, last_iter=last_iter
    )
    train_loader = DataLoader(
        train_dataset, batch_size=config.batch_size, shuffle=False,
        num_workers=config.workers, pin_memory=True, sampler=train_sampler
    )

    # validation augmentation
    trans = albumentations.Compose([
        Resize(config.augmentation.input_size, config.augmentation.input_size),
        Normalize(mean=model_mean, std=model_std),
        ToTensorV2()
    ])
    val_multi_loader = []
    if args.val_source != '':
        for dataset_idx in range(len(args.val_source)):
            val_dataset = FaceDataset(
                args.val_root[dataset_idx],
                args.val_source[dataset_idx],
                transform=trans,
                output_index=True,
                resize=config.augmentation.input_size,
                image_format=config.get('image_format', None),
                bgr=config.augmentation.get('bgr', False)
            )
            val_sampler = DistributedSampler(val_dataset, round_up=False)
            val_loader = DataLoader(
                val_dataset, batch_size=config.batch_size, shuffle=False,
                num_workers=config.workers, pin_memory=True, sampler=val_sampler
            )
            val_multi_loader.append(val_loader)

    config.lr_scheduler['optimizer'] = optimizer
    config.lr_scheduler['last_iter'] = last_iter
    lr_scheduler = get_scheduler(config.lr_scheduler)
    
    if rank == 0:
        mkdir(args.save_path)
            
        mkdir(args.save_path_dated)
        tb_logger = SummaryWriter(args.save_path_dated)
        
        logger = create_logger('global_logger', args.save_path_dated + '-log.txt')
        logger.info('{}'.format(args))
        logger.info(model)
        logger.info(parameters_string(model))
        logger.info('len(train dataset) = %d'%len(train_loader.dataset))
        for dataset_idx in range(len(val_multi_loader)):
            logger.info('len(val%d dataset) = %d'%(dataset_idx, len(val_multi_loader[dataset_idx].dataset)))
        
        mkdir(args.save_path_dated + '/saves')
    else:
        tb_logger = None
        
    positive_weight = config.get('positive_weight', 0.5)
    weight = torch.tensor([1.-positive_weight, positive_weight]) * 2.
    if rank == 0:
        logger.info('using class weights: {}'.format(weight.tolist()))

    criterion = nn.CrossEntropyLoss(weight=weight).cuda()

    if args.evaluate:
        if args.evaluate_path:
            all_ckpt = get_all_checkpoint(args.evaluate_path, args.range_list, rank)

            for ckpt in all_ckpt:
                if rank == 0:
                    logger.info('Testing ckpt: ' + ckpt)
                last_iter = -1
                _, last_iter = load_state(ckpt, model, optimizer=optimizer)
                for dataset_idx in range(len(val_multi_loader)):
                    validate(dataset_idx, val_multi_loader[dataset_idx], model, criterion, tb_logger, 
                             curr_step=last_iter, save_softmax=True)
        else:
            for dataset_idx in range(len(val_multi_loader)):
                validate(dataset_idx, val_multi_loader[dataset_idx], model, criterion, tb_logger, 
                         curr_step=last_iter, save_softmax=True)

        return

    train(train_loader, val_multi_loader, model, criterion, optimizer, 
          lr_scheduler, last_iter+1, tb_logger)
    return

def train(train_loader, val_multi_loader, model, criterion, optimizer, 
          lr_scheduler, start_iter, tb_logger):

    global best_loss

    batch_time = AverageMeter(config.print_freq)
    fw_time = AverageMeter(config.print_freq)
    bp_time = AverageMeter(config.print_freq)
    sy_time = AverageMeter(config.print_freq)
    step_time = AverageMeter(config.print_freq)
    data_time = AverageMeter(config.print_freq)
    losses = AverageMeter(config.print_freq)
    top1 = AverageMeter(config.print_freq)
    top2 = AverageMeter(config.print_freq)

    # switch to train mode
    model.train()

    world_size = dist.get_world_size()
    rank = dist.get_rank()

    logger = logging.getLogger('global_logger')

    end = time.time()

    for i, (input, target) in enumerate(train_loader):
        curr_step = start_iter + i
        lr_scheduler.step(curr_step)
        current_lr = lr_scheduler.get_lr()[0]

        # measure data loading time
        data_time.update(time.time() - end)

        # transfer input to gpu
        target = target.cuda()
        input = input.cuda()

        # forward
        output = model(input)
        loss = criterion(output, target) / world_size

        # measure accuracy and record loss
        prec1, prec2 = accuracy(output, target, topk=(1, 2))

        reduced_loss = loss.clone()
        reduced_prec1 = prec1.clone() / world_size
        reduced_prec2 = prec2.clone() / world_size

        dist.all_reduce(reduced_loss)
        dist.all_reduce(reduced_prec1)
        dist.all_reduce(reduced_prec2)

        losses.update(reduced_loss.item())
        top1.update(reduced_prec1.item())
        top2.update(reduced_prec2.item())

        # backward
        optimizer.zero_grad()

        loss.backward()
        optimizer.step()

        # measure elapsed time
        batch_time.update(time.time() - end)
        
        if curr_step % config.print_freq == 0 and rank == 0:
            total_batch_size = world_size * args.batch_size
            epoch = (curr_step * total_batch_size) // len(train_loader.dataset)
            tb_logger.add_scalar('loss_train', losses.avg, curr_step)
            tb_logger.add_scalar('acc1_train', top1.avg, curr_step)
            tb_logger.add_scalar('acc2_train', top2.avg, curr_step)
            tb_logger.add_scalar('lr', current_lr, curr_step)
            logger.info('Iter: [{0}/{1}]\t'
                  'Epoch: {2}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'LR {lr:.4f}'.format(
                   curr_step, len(train_loader), epoch, batch_time=batch_time,
                   data_time=data_time, loss=losses, top1=top1, lr=current_lr))

        if curr_step > 0 and curr_step % config.val_freq == 0:
            if not args.no_val:
                total_loss = 0
                for dataset_idx in range(len(val_multi_loader)):
                    val_loss, prec1, prec2 = validate(dataset_idx, val_multi_loader[dataset_idx], model, criterion, 
                                                      tb_logger, curr_step=curr_step, save_softmax=True)

                    total_loss += val_loss
                # average loss over multiple validation sets
                if len(val_multi_loader) > 0:
                    loss = total_loss / len(val_multi_loader)
            else:
                loss = 1e9
           
            if rank == 0:
                # remember best video logloss recorded at rank 0 and save checkpoint
                is_best = loss < best_loss
                best_loss = min(loss, best_loss)
                save_checkpoint({
                    'step': curr_step,
                    'arch': config.model.arch,
                    'state_dict': model.state_dict(),
                    'best_loss': best_loss,
                    'optimizer' : optimizer.state_dict(),
                }, is_best, args.save_path_dated +'/ckpt')

        end = time.time()

def validate(dataset_idx, val_loader, model, criterion, tb_logger, 
             curr_step=None, save_softmax=False):
    batch_time = AverageMeter(0)
    losses = AverageMeter(0)
    top1 = AverageMeter(0)
    top2 = AverageMeter(0)

    # switch to evaluate mode
    model.eval()

    rank = dist.get_rank()
    world_size = dist.get_world_size()

    logger = logging.getLogger('global_logger')

    end = time.time()
    if save_softmax:
        output_dict = {}
        mkdir(args.save_path_dated + '/transient')
    with torch.no_grad():
        for i, (input, target, index) in enumerate(val_loader):
            input = input.cuda()
            target = target.cuda()
            # compute output
            output = model(input)

            # measure accuracy and record loss
            loss = criterion(output, target) # loss should not be scaled here, it's reduced later!
            
            if save_softmax:
                output = torch.nn.functional.softmax(output, dim=1)
                for idx in range(len(index)):
                    cur_index = index[idx].item()
                    meta, relative_idx = val_loader.dataset.lookup_meta(cur_index)
                    name, start, stride, count, label = meta
                    frame = start + stride * relative_idx
                    output_dict[cur_index] = {'filename': name,
                                              'prob': output[idx][1].item(),
                                              'label': label,
                                              'frame': frame}
                    
            prec1, prec2 = accuracy(output.data, target, topk=(1, 2))

            num = input.size(0)
            losses.update(loss.item(), num)
            top1.update(prec1.item(), num)
            top2.update(prec2.item(), num)

            # measure elapsed time
            batch_time.update(time.time() - end)
            end = time.time()

            if i % config.print_freq == 0 and rank == 0:
                logger.info('Test: [{0}/{1}]\tTime {batch_time.val:.3f} ({batch_time.avg:.3f})'.format(
                       i, len(val_loader), batch_time=batch_time))

    # gather final results
    total_num = torch.Tensor([losses.count]).cuda()
    loss_sum = torch.Tensor([losses.avg*losses.count]).cuda()
    top1_sum = torch.Tensor([top1.avg*top1.count]).cuda()
    top2_sum = torch.Tensor([top2.avg*top2.count]).cuda()
    dist.all_reduce(total_num)
    dist.all_reduce(loss_sum)
    dist.all_reduce(top1_sum)
    dist.all_reduce(top2_sum)
    final_loss = loss_sum.item()/total_num.item()
    final_top1 = top1_sum.item()/total_num.item()
    final_top2 = top2_sum.item()/total_num.item()
    if rank == 0:
        logger.info(' * Prec@1 {:.3f}\tPrec@2 {:.3f}\tLoss {:.6f}\ttotal_num={}'.format(final_top1, final_top2, final_loss, total_num.item()))
        
    # combine softmax values from all GPUs
    if save_softmax:
        with open(args.save_path_dated + '/transient/tmp%d.pkl'%rank, 'wb') as f:
            pickle.dump(output_dict, f)
        
        # block until all processes have finished saving their results, then proceed combining
        dist.barrier()
        if rank == 0:
            score_list = {}
            for idx in range(world_size):
                with open(args.save_path_dated + '/transient/tmp%d.pkl'%idx, "rb") as f:
                    score = pickle.load(f)
                    score_list.update(score)

            with open(args.save_path_dated + '/saves/val%d_score_%d.pkl'%(dataset_idx, curr_step), 'wb') as f:
                pickle.dump(score_list, f)

            video_score_list = {}
            for idx in score_list:
                score = score_list[idx]
                vid = '/'.join(score['filename'].split('/')[:-1])
                
                prob = score['prob']
                
                if vid not in video_score_list:
                    video_score_list[vid] = {'prob': prob, 'count': 1, 'label': score['label']}
                else:
                    video_score_list[vid]['prob'] += prob
                    video_score_list[vid]['count'] += 1
            
            logloss, acc1 = 0., 0.
            cls_logloss, cls_acc1, cls_cnt = [0., 0.], [0., 0.], [0, 0]
            for vid in video_score_list:
                video_score_list[vid]['prob'] /= video_score_list[vid]['count']
                
                prob = video_score_list[vid]['prob']
                label = video_score_list[vid]['label']
                cur_logloss = -(label * math.log(prob + 1e-15) + (1 - label) * math.log(1 - prob + 1e-15))
                cur_acc1 = ((prob >= 0.5) == label)
                
                logloss += cur_logloss
                acc1 += cur_acc1
                cls_cnt[label] += 1
                cls_logloss[label] += cur_logloss
                cls_acc1[label] += cur_acc1
                
            logloss /= len(video_score_list)
            acc1 /= len(video_score_list) / 100.
            if cls_cnt[0] > 0:
                cls_logloss[0] /= cls_cnt[0]
                cls_acc1[0] /= cls_cnt[0] / 100.
            if cls_cnt[1] > 0:
                cls_logloss[1] /= cls_cnt[1]
                cls_acc1[1] /= cls_cnt[1] / 100.
            balanced_logloss = (cls_logloss[0] + cls_logloss[1]) / 2.
            balanced_acc1 = (cls_acc1[0] + cls_acc1[1]) / 2.
            
            logger.info(' * logLoss {:.6f}\tAcc1 {:.4f}%\ttotal_videos={}'.format(logloss, acc1, len(video_score_list)))
            logger.info(' * REAL logLoss {:.6f}\tAcc1 {:.4f}%\ttotal_videos={}'.format(cls_logloss[0], cls_acc1[0], cls_cnt[0]))
            logger.info(' * FAKE logLoss {:.6f}\tAcc1 {:.4f}%\ttotal_videos={}'.format(cls_logloss[1], cls_acc1[1], cls_cnt[1]))
            logger.info(' * Balanced logLoss {:.6f}\tAcc1 {:.4f}%\ttotal_videos={}'.format(balanced_logloss, balanced_acc1, len(video_score_list)))
            with open(args.save_path_dated + '/saves/val%d_video_score_%d.pkl'%(dataset_idx, curr_step), 'wb') as f:
                pickle.dump(video_score_list, f)
            video_score_csv = [[key.split('/')[-1] + '.mp4', video_score_list[key]['prob']] for key in video_score_list]
            video_score_csv.sort(key=lambda item: item[0])
            with open(args.save_path_dated + '/saves/val%d_video_score_%d.csv' % (dataset_idx, curr_step), 'w') as f:
                for score in video_score_csv:
                    f.write(score[0] + ',' + str(score[1]) + '\n')
            if not tb_logger is None:
                tb_logger.add_scalar('loss_val%d'%dataset_idx, final_loss, curr_step)
                tb_logger.add_scalar('acc1_val%d'%dataset_idx, final_top1, curr_step)
                tb_logger.add_scalar('video_logloss_val%d'%dataset_idx, logloss, curr_step)
                tb_logger.add_scalar('video_REAL_logloss_val%d'%dataset_idx, cls_logloss[0], curr_step)
                tb_logger.add_scalar('video_FAKE_logloss_val%d'%dataset_idx, cls_logloss[1], curr_step)
                tb_logger.add_scalar('video_balanced_logloss_val%d'%dataset_idx, balanced_logloss, curr_step)
                tb_logger.add_scalar('video_acc1_val%d'%dataset_idx, acc1, curr_step)
                tb_logger.add_scalar('video_REAL_acc1_val%d'%dataset_idx, cls_acc1[0], curr_step)
                tb_logger.add_scalar('video_FAKE_acc1_val%d'%dataset_idx, cls_acc1[1], curr_step)
                tb_logger.add_scalar('video_balanced_acc1_val%d'%dataset_idx, balanced_acc1, curr_step)
            
            # record video logloss in rank 0
            final_loss = balanced_logloss
    dist.barrier()
    model.train()
    
    return final_loss, final_top1, final_top2


if __name__ == '__main__':
    main()
