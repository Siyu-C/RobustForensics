from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
from PIL import Image
import bisect
import random
import torch
import torch.distributed as dist


def pil_loader(img_str):
    with open(img_str, 'rb') as f:
        with Image.open(f) as img:
            img = img.convert('RGB')
    return img


class VideoDataset(Dataset):
    def __init__(self, root_dir, source, spatial_transform=None, 
                 temporal_transform=None, output_index=False, image_format=None):
        super(VideoDataset, self).__init__()
        
        self.root_dir = root_dir
        self.spatial_transform = spatial_transform
        self.temporal_transform = temporal_transform
        
        self.rank = dist.get_rank()
        
        self.output_index = output_index
        # If not None, subsitute default image format (frame%d.png) with self.image_format
        # Example: frame%d_face1.jpg
        self.image_format = image_format
        
        with open(source) as f:
            lines = f.readlines()

        if self.rank == 0:
            print("building dataset from %s"%source)

        # meta format:
        # [relative path] [start frame id] [sample stride] [# frames] [label]
        # Example:
        # dfdc_train_part_32/qepoibkeoq/frame%d.png 1 30 10 1
        self.metas = []
        for line in lines:
            path, start, stride, count, cls = line.rstrip().split()
            if self.image_format is not None:
                paths = path.split('/')
                paths[-1] = self.image_format
                path = '/'.join(paths)
            # make sure video datum has continuous frames
            assert int(stride) == 1
            self.metas.append((path, int(start), int(stride), int(count), int(cls)))
        self.num = len(self.metas)
 
    def __len__(self):
        return self.num
            
    def lookup_meta(self, idx):
        return self.metas[idx], 0
 
    def __getitem__(self, idx):
        meta, _ = self.lookup_meta(idx)
        path, start, _, count, cls = meta
        
        frame_indices = list(range(start, start+count))
        if self.temporal_transform is not None:
            frame_indices = self.temporal_transform(frame_indices)
        
        clip = []
        for i in frame_indices:
            image_path = os.path.join(self.root_dir, path%i)
            try:
                img = pil_loader(image_path)
            except Exception as e:
                print(e)
                print("[ERROR] something wrong when loading the image (probably missing): {}".format(image_path))
                
                # random image
                img = np.random.randint(0, 255, (224, 224, 3), dtype=np.uint8)
                img = Image.fromarray(img)
            clip.append(img)
            
        if self.spatial_transform is not None:
            self.spatial_transform.randomize_parameters()
            clip = [self.spatial_transform(img) for img in clip]
        clip = torch.stack(clip, 0).permute(1, 0, 2, 3)
        
        if self.output_index:
            return clip, cls, idx
        else:
            return clip, cls
