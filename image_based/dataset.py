from torch.utils.data import DataLoader, Dataset
import numpy as np
import os
import cv2
import bisect
import random
import torch.distributed as dist


def cv2_loader(img_str):
    return cv2.imread(img_str, cv2.IMREAD_COLOR)


class FaceDataset(Dataset):
    def __init__(self, root_dir, source, transform=None,
                 output_index=False, resize=299, image_format=None,
                 random_frame=False, bgr=False):
        super(FaceDataset, self).__init__()

        self.root_dir = root_dir
        self.transform = transform
        self.rank = dist.get_rank()
        self.world_size = dist.get_world_size()
        self.bgr = bgr
        self.output_index = output_index
        self.resize = resize
        # If not None, subsitute default image format (frame%d.png) with self.image_format
        # Example: frame%d_face1.jpg
        self.image_format = image_format
        self.random_frame = random_frame

        with open(source) as f:
            lines = f.readlines()

        if self.rank == 0:
            print("building dataset from %s"%source)

        # meta format:
        # [relative path] [start frame id] [sample stride] [# frames] [label]
        # Example:
        # dfdc_train_part_32/qepoibkeoq/frame%d.png 1 30 10 1
        self.metas = []
        self.num, self.lookup_table = 0, [0]
        for line in lines:
            path, start, stride, count, cls = line.rstrip().split()
            if self.image_format is not None:
                paths = path.split('/')
                paths[-1] = self.image_format
                path = '/'.join(paths)
            if int(count) <= 0:
                continue
            self.metas.append((path, int(start), int(stride), int(count), int(cls)))
            self.num += int(count)
            self.lookup_table.append(self.num)

    def __len__(self):
        if self.random_frame:
            return len(self.metas)
        return self.num

    def lookup_meta(self, idx):
        if self.random_frame:
            meta = self.metas[idx]
            return meta, random.randint(0, meta[3] - 1)
        meta_idx = bisect.bisect_right(self.lookup_table, idx) - 1
        return self.metas[meta_idx], idx - self.lookup_table[meta_idx]

    def __getitem__(self, idx):
        meta, relative_idx = self.lookup_meta(idx)
        path, start, stride, count, cls = meta
        filename = os.path.join(self.root_dir, path%(start + stride * relative_idx))

        try:
            img = cv2_loader(filename)
            if not self.bgr:
                img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        except Exception as e:
            print(e)
            print("[ERROR] Image loading failed! Location: {}".format(filename))
            if self.mix:
                b = np.random.randint(0, 255, (self.resize, self.resize), dtype=np.uint8)
                g = np.random.randint(0, 255, (self.resize, self.resize), dtype=np.uint8)
                r = np.random.randint(0, 255, (self.resize, self.resize), dtype=np.uint8)

                img = cv2.merge([b, g, r])
            else:
                # random image with resolution self.resize * self.resize
                img = np.random.randint(0, 255, (self.resize, self.resize, 3), dtype=np.uint8)
                img = Image.fromarray(img)
                # set class to FAKE
                cls = 1
                
        img = self.transform(image=img)['image']

        if self.output_index:
            return img, cls, idx
        else:
            return img, cls
