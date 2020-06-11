import random
import collections
import numpy as np
import torch
from PIL import Image, ImageOps
try:
    import accimage
except ImportError:
    accimage = None
import cv2
import albumentations.augmentations.functional as F


class Compose(object):
    """Composes several transforms together.
    Args:
        transforms (list of ``Transform`` objects): list of transforms to compose.
    Example:
        >>> transforms.Compose([
        >>>     transforms.CenterCrop(10),
        >>>     transforms.ToTensor(),
        >>> ])
    """

    def __init__(self, transforms):
        self.transforms = transforms

    def __call__(self, img):
        for t in self.transforms:
            img = t(img)
        return img

    def randomize_parameters(self):
        for t in self.transforms:
            t.randomize_parameters()


class ToTensor(object):
    """Convert a ``PIL.Image`` or ``numpy.ndarray`` to tensor.
    Converts a PIL.Image or numpy.ndarray (H x W x C) in the range
    [0, 255] to a torch.FloatTensor of shape (C x H x W) in the range [0.0, 1.0].
    """

    def __init__(self, norm_value=255):
        self.norm_value = norm_value

    def __call__(self, pic):
        """
        Args:
            pic (PIL.Image or numpy.ndarray): Image to be converted to tensor.
        Returns:
            Tensor: Converted image.
        """
        if isinstance(pic, np.ndarray):
            # handle numpy array
            img = torch.from_numpy(pic.transpose((2, 0, 1)))
            # backward compatibility
            return img.float().div(self.norm_value)

        if accimage is not None and isinstance(pic, accimage.Image):
            nppic = np.zeros(
                [pic.channels, pic.height, pic.width], dtype=np.float32)
            pic.copyto(nppic)
            return torch.from_numpy(nppic)

        # handle PIL Image
        if pic.mode == 'I':
            img = torch.from_numpy(np.array(pic, np.int32, copy=False))
        elif pic.mode == 'I;16':
            img = torch.from_numpy(np.array(pic, np.int16, copy=False))
        else:
            img = torch.ByteTensor(torch.ByteStorage.from_buffer(pic.tobytes()))
        # PIL image mode: 1, L, P, I, F, RGB, YCbCr, RGBA, CMYK
        if pic.mode == 'YCbCr':
            nchannel = 3
        elif pic.mode == 'I;16':
            nchannel = 1
        else:
            nchannel = len(pic.mode)
        img = img.view(pic.size[1], pic.size[0], nchannel)
        # put it from HWC to CHW format
        # yikes, this transpose takes 80% of the loading time/CPU
        img = img.transpose(0, 1).transpose(0, 2).contiguous()
        if isinstance(img, torch.ByteTensor):
            return img.float().div(self.norm_value)
        else:
            return img

    def randomize_parameters(self):
        pass


class Normalize(object):
    """Normalize an tensor image with mean and standard deviation.
    Given mean: (R, G, B) and std: (R, G, B),
    will normalize each channel of the torch.*Tensor, i.e.
    channel = (channel - mean) / std
    Args:
        mean (sequence): Sequence of means for R, G, B channels respecitvely.
        std (sequence): Sequence of standard deviations for R, G, B channels
            respecitvely.
    """

    def __init__(self, mean, std):
        self.mean = mean
        self.std = std

    def __call__(self, tensor):
        """
        Args:
            tensor (Tensor): Tensor image of size (C, H, W) to be normalized.
        Returns:
            Tensor: Normalized image.
        """
        # TODO: make efficient
        for t, m, s in zip(tensor, self.mean, self.std):
            t.sub_(m).div_(s)
        return tensor

    def randomize_parameters(self):
        pass


class Scale(object):
    """Rescale the input PIL.Image to the given size.
    Args:
        size (sequence or int): Desired output size. If size is a sequence like
            (w, h), output size will be matched to this. If size is an int,
            smaller edge of the image will be matched to this number.
            i.e, if height > width, then image will be rescaled to
            (size * height / width, size)
        interpolation (int, optional): Desired interpolation. Default is
            ``PIL.Image.BILINEAR``
    """

    def __init__(self, size, interpolation=Image.BILINEAR):
        assert isinstance(size,
                          int) or (isinstance(size, collections.Iterable) and
                                   len(size) == 2)
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be scaled.
        Returns:
            PIL.Image: Rescaled image.
        """
        if isinstance(self.size, int):
            w, h = img.size
            if (w <= h and w == self.size) or (h <= w and h == self.size):
                return img
            if w < h:
                ow = self.size
                oh = int(self.size * h / w)
                return img.resize((ow, oh), self.interpolation)
            else:
                oh = self.size
                ow = int(self.size * w / h)
                return img.resize((ow, oh), self.interpolation)
        else:
            return img.resize(self.size, self.interpolation)

    def randomize_parameters(self):
        pass


class RandomHorizontalFlip(object):
    """Horizontally flip the given PIL.Image randomly with a probability of 0.5."""

    def __call__(self, img):
        """
        Args:
            img (PIL.Image): Image to be flipped.
        Returns:
            PIL.Image: Randomly flipped image.
        """
        if self.p < 0.5:
            return img.transpose(Image.FLIP_LEFT_RIGHT)
        return img

    def randomize_parameters(self):
        self.p = random.random()

        
class ScaleJitteringRandomCrop(object):

    def __init__(self, min_scale, max_scale, size, interpolation=Image.BILINEAR):
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.size = size
        self.interpolation = interpolation

    def __call__(self, img):
        min_length = min(img.size[0], img.size[1])
        jitter_rate = float(self.scale) / min_length

        image_width = int(jitter_rate * img.size[0])
        image_height = int(jitter_rate * img.size[1])
        img = img.resize((image_width, image_height), self.interpolation)

        x1 = self.tl_x * (image_width - self.size)
        y1 = self.tl_y * (image_height - self.size)
        x2 = x1 + self.size
        y2 = y1 + self.size

        return img.crop((x1, y1, x2, y2))

    def randomize_parameters(self):
        self.scale = random.randint(self.min_scale, self.max_scale)
        self.tl_x = random.random()
        self.tl_y = random.random()
        
        
class RandomBrightnessContrast(object):
    """Randomly change brightness and contrast of the input image.

    Args:
        brightness_limit ((float, float) or float): factor range for changing brightness.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        contrast_limit ((float, float) or float): factor range for changing contrast.
            If limit is a single float, the range will be (-limit, limit). Default: (-0.2, 0.2).
        brightness_by_max (Boolean): If True adjust contrast by image dtype maximum,
            else adjust contrast by image mean.
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """

    def __init__(self, brightness_limit=0.2, contrast_limit=0.2, brightness_by_max=True, always_apply=False, p=0.5):
        self.brightness_limit = (-brightness_limit, brightness_limit)
        self.contrast_limit = (-contrast_limit, contrast_limit)
        self.brightness_by_max = brightness_by_max
        
        self.always_apply = always_apply
        self.p = p

    def __call__(self, img):
        if self.always_apply or self.apply < self.p:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return F.brightness_contrast_adjust(img, self.alpha, self.beta, self.brightness_by_max)
        return img

    def randomize_parameters(self):
        self.alpha = 1.0 + random.uniform(self.contrast_limit[0], self.contrast_limit[1])
        self.beta = 0.0 + random.uniform(self.brightness_limit[0], self.brightness_limit[1])
        self.apply = random.random()
        
        
class ImageCompression(object):
    """Decrease Jpeg, WebP compression of an image.

    Args:
        quality_lower (float): lower bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        quality_upper (float): upper bound on the image quality.
                               Should be in [0, 100] range for jpeg and [1, 100] for webp.
        compression_type (ImageCompressionType): should be JPEG or WEBP.
            Default: JPEG

    Targets:
        image

    Image types:
        uint8, float32
    """

    JPEG = 0
    WEBP = 1

    def __init__(
        self,
        quality_lower=99,
        quality_upper=100,
        compression_type=JPEG,
        always_apply=False,
        p=0.5,
    ):
        self.compression_type = compression_type
        low_thresh_quality_assert = 0

        if self.compression_type == ImageCompression.WEBP:
            low_thresh_quality_assert = 1

        assert low_thresh_quality_assert <= quality_lower <= 100
        assert low_thresh_quality_assert <= quality_upper <= 100

        self.quality_lower = quality_lower
        self.quality_upper = quality_upper
        
        self.always_apply = always_apply
        self.p = p

    def __call__(self, img):
        if self.always_apply or self.apply < self.p:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return F.image_compression(img, self.quality, self.image_type)
        return img

    def randomize_parameters(self):
        self.image_type = ".jpg"

        if self.compression_type == ImageCompression.WEBP:
            self.image_type = ".webp"

        self.quality = random.randint(self.quality_lower, self.quality_upper)
        self.apply = random.random()

        
class MotionBlur(object):
    """Apply motion blur to the input image using a random-sized kernel.

    Args:
        blur_limit (int): maximum kernel size for blurring the input image.
            Should be in range [3, inf). Default: (3, 7).
        p (float): probability of applying the transform. Default: 0.5.

    Targets:
        image

    Image types:
        uint8, float32
    """
    
    def __init__(self, blur_limit=7, always_apply=False, p=0.5):
        assert blur_limit >= 3
        self.blur_limit = (3, blur_limit)
        
        self.always_apply = always_apply
        self.p = p

    def __call__(self, img):
        if self.always_apply or self.apply < self.p:
            if not isinstance(img, np.ndarray):
                img = np.array(img)
            return F.motion_blur(img, kernel=self.kernel)
        return img

    def randomize_parameters(self):
        ksize = random.choice(np.arange(self.blur_limit[0], self.blur_limit[1] + 1, 2))
        assert ksize > 2
        kernel = np.zeros((ksize, ksize), dtype=np.uint8)
        xs, xe = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        if xs == xe:
            ys, ye = random.sample(range(ksize), 2)
        else:
            ys, ye = random.randint(0, ksize - 1), random.randint(0, ksize - 1)
        cv2.line(kernel, (xs, ys), (xe, ye), 1, thickness=1)

        # Normalize kernel
        self.kernel = kernel.astype(np.float32) / np.sum(kernel)
        self.apply = random.random()
    