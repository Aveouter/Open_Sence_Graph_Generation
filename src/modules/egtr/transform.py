"""
DETR-style data augmentation transforms.
Borrowed from: https://github.com/facebookresearch/detr/blob/main/datasets/transforms.py
Minimal implementation — only needed for DeformableDetrFeatureExtractorWithAugmentor.
"""

import random
import torch
import torchvision.transforms as T
import torchvision.transforms.functional as F
from PIL import Image


def crop(image, target, region):
    cropped_image = F.crop(image, *region)
    target = target.copy()
    i, j, h, w = region
    size = cropped_image.size
    target["size"] = torch.tensor([size[1], size[0]])
    
    fields = ["labels", "area", "iscrowd"]
    if "boxes" in target:
        boxes = target["boxes"]
        max_size = torch.as_tensor([w, h], dtype=torch.float32)
        cropped_boxes = boxes - torch.as_tensor([j, i, j, i])
        cropped_boxes = torch.min(cropped_boxes.reshape(-1, 2, 2), max_size)
        cropped_boxes = cropped_boxes.clamp(min=0)
        target["boxes"] = cropped_boxes.reshape(-1, 4)
        fields.append("boxes")
    
    if "masks" in target:
        target["masks"] = target["masks"][:, i:i + h, j:j + w]
        fields.append("masks")
    
    if "boxes" in target and "labels" in target:
        kept = (target["boxes"][:, 2] > target["boxes"][:, 0]) & (target["boxes"][:, 3] > target["boxes"][:, 1])
        for f in fields:
            target[f] = target[f][kept]
    
    return cropped_image, target


def hflip(image, target):
    flipped_image = F.hflip(image)
    w, h = image.size
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        boxes = boxes[:, [2, 1, 0, 3]] * torch.as_tensor([-1, 1, -1, 1]) + torch.as_tensor([w, 0, w, 0])
        target["boxes"] = boxes
    if "masks" in target:
        target["masks"] = target["masks"].flip(-1)
    return flipped_image, target


def resize(image, target, size, max_size=None):
    def get_size_with_aspect_ratio(image_size, size, max_size=None):
        w, h = image_size
        if max_size is not None:
            min_original_size = float(min((w, h)))
            max_original_size = float(max((w, h)))
            if max_original_size / min_original_size * size > max_size:
                size = int(round(max_size * min_original_size / max_original_size))
        if (w <= h and w == size) or (h <= w and h == size):
            return (h, w)
        if w < h:
            ow = size
            oh = int(size * h / w)
        else:
            oh = size
            ow = int(size * w / h)
        return (oh, ow)
    
    def get_size(image_size, size, max_size=None):
        if isinstance(size, (list, tuple)):
            return size[::-1]
        else:
            return get_size_with_aspect_ratio(image_size, size, max_size)
    
    size = get_size(image.size, size, max_size)
    rescaled_image = F.resize(image, size)
    
    if target is None:
        return rescaled_image, None
    
    ratios = tuple(float(s) / float(s_orig) for s, s_orig in zip(rescaled_image.size, image.size))
    ratio_width, ratio_height = ratios
    
    target = target.copy()
    if "boxes" in target:
        boxes = target["boxes"]
        scaled_boxes = boxes * torch.as_tensor([ratio_width, ratio_height, ratio_width, ratio_height])
        target["boxes"] = scaled_boxes
    
    if "area" in target:
        area = target["area"]
        scaled_area = area * (ratio_width * ratio_height)
        target["area"] = scaled_area
    
    h, w = size
    target["size"] = torch.tensor([h, w])
    
    if "masks" in target:
        target["masks"] = F.resize(target["masks"][:, None], size, interpolation=F.InterpolationMode.NEAREST)[:, 0]
    
    return rescaled_image, target


class RandomResize:
    def __init__(self, sizes, max_size=None):
        assert isinstance(sizes, (list, tuple))
        self.sizes = sizes
        self.max_size = max_size
    
    def __call__(self, img, target):
        size = random.choice(self.sizes)
        return resize(img, target, size, self.max_size)


class RandomHorizontalFlip:
    def __init__(self, p=0.5):
        self.p = p
    
    def __call__(self, img, target):
        if random.random() < self.p:
            return hflip(img, target)
        return img, target


class RandomSizeCrop:
    def __init__(self, min_size: int, max_size: int):
        self.min_size = min_size
        self.max_size = max_size
    
    def __call__(self, img, target):
        w = random.randint(self.min_size, min(img.width, self.max_size))
        h = random.randint(self.min_size, min(img.height, self.max_size))
        region = T.RandomCrop.get_params(img, [h, w])
        return crop(img, target, region)


class RandomSelect:
    def __init__(self, transforms1, transforms2, p=0.5):
        self.transforms1 = transforms1
        self.transforms2 = transforms2
        self.p = p
    
    def __call__(self, img, target):
        if random.random() < self.p:
            return self.transforms1(img, target)
        return self.transforms2(img, target)


class Compose:
    def __init__(self, transforms):
        self.transforms = transforms
    
    def __call__(self, image, target):
        for t in self.transforms:
            image, target = t(image, target)
        return image, target
    
    def __repr__(self):
        format_string = self.__class__.__name__ + "("
        for t in self.transforms:
            format_string += "\n"
            format_string += f"    {t}"
        format_string += "\n)"
        return format_string
