import random
import math

from PIL import Image
from torchvision.transforms import (
    ToTensor, Normalize, Compose, Resize, CenterCrop, RandomCrop,
    RandomHorizontalFlip)

import torchvision.transforms as trf

try:
    import accimage
except ImportError:
    accimage = None


class RandomSizedCrop:
    """Random crop the given PIL.Image to a random size
    of the original size and and a random aspect ratio
    of the original aspect ratio.
    size: size of the smaller edge
    interpolation: Default: PIL.Image.BILINEAR
    """

    def __init__(self, size, interpolation=Image.BILINEAR,
                 min_aspect=4/5, max_aspect=5/4,
                 min_area=0.25, max_area=1):
        self.size = size
        self.interpolation = interpolation
        self.min_aspect = min_aspect
        self.max_aspect = max_aspect
        self.min_area = min_area
        self.max_area = max_area

    def __call__(self, img):
        for attempt in range(10):
            area = img.size[0] * img.size[1]
            target_area = random.uniform(self.min_area, self.max_area) * area
            aspect_ratio = random.uniform(self.min_aspect, self.max_aspect)

            w = int(round(math.sqrt(target_area * aspect_ratio)))
            h = int(round(math.sqrt(target_area / aspect_ratio)))

            if random.random() < 0.5:
                w, h = h, w

            if w <= img.size[0] and h <= img.size[1]:
                x1 = random.randint(0, img.size[0] - w)
                y1 = random.randint(0, img.size[1] - h)

                img = img.crop((x1, y1, x1 + w, y1 + h))
                assert(img.size == (w, h))

                return img.resize((self.size, self.size), self.interpolation)

        # Fallback
        scale = Resize(self.size, interpolation=self.interpolation)
        crop = CenterCrop(self.size)
        return crop(scale(img))


class MyResize(Resize):
    def __init__(self, size, interpolation=Image.BILINEAR):
        Resize.__init__(self, size=size, interpolation=interpolation)
    def __call__(self, img):
        """
        Args:
            img (PIL Image): Image to be scaled.

        Returns:
            PIL Image: Rescaled image.
        """
        def _is_pil_image(img):
            if accimage is not None:
                return isinstance(img, (Image.Image, accimage.Image))
            else:
                return isinstance(img, Image.Image)
        def resize(img, size, interpolation=Image.BILINEAR):
            if not _is_pil_image(img):
                raise TypeError('img should be PIL Image. Got {}'.format(type(img)))
            # if not (isinstance(size, int) or (isinstance(size, Iterable) and len(size) == 2)):
            #     raise TypeError('Got inappropriate size arg: {}'.format(size))

            if isinstance(size, int):
                w, h = img.size
                if (w <= h and h == size) or (h <= w and w == size):
                    return img
                if w < h:
                    oh = size
                    ow = int(size * w / h)
                    return img.resize((ow, oh), interpolation)
                else:
                    ow = size
                    oh = int(size * h / w)
                    # out = img.resize((ow, oh), interpolation)
                    return img.resize((ow, oh), interpolation)
            else:
                return img.resize(size[::-1], interpolation)
        return resize(img, self.size, self.interpolation)


# train_transform = Compose([
#     trf.RandomApply([RandomHorizontalFlip(p=0.4),
#                      trf.RandomVerticalFlip(p=0.2),
#                      trf.RandomRotation(45),
#                      trf.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)], p=0.9),
#     RandomSizedCrop(size=288, min_area=0.8),
# ])
# trf.RandomChoice([RandomCrop(288),Resize(288)]),
# train_transform = Compose([
#     trf.RandomApply([trf.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)], p=0.8),
#     RandomCrop(288),
#     RandomHorizontalFlip(),
# ])


train_transform = Compose([
    trf.RandomApply([trf.ColorJitter(brightness=0.2, contrast=0.2, hue=0.2)], p=0.8),
    trf.RandomApply([trf.RandomRotation(10)], p=0.8),
    # trf.Resize(size=256),
    MyResize(size=288),
    RandomCrop(size=288, pad_if_needed=True, padding=0),
    RandomHorizontalFlip(),
])




# test_transform = Compose([
#     # trf.RandomApply([RandomHorizontalFlip(p=0.4),
#     #                  trf.RandomVerticalFlip(p=0.2),
#     #                  trf.RandomRotation(45),
#     #                  trf.ColorJitter(brightness=0.1, contrast=0.1, hue=0.1)], p=0.9),
#     RandomSizedCrop(size=288, min_area=0.8),
# ])
test_transform = Compose([
    MyResize(size=288),
    # trf.Resize(size=256),
    RandomCrop(size=288,pad_if_needed=True,padding=0),
    RandomHorizontalFlip(),
])



tensor_transform = Compose([
    ToTensor(),
    Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
])


'''
import torch
from torch.utils.data import DataLoader
from torchvision import transforms
import torchvision.datasets as datasets
import matplotlib.pyplot as plt

# a simple custom collate function, just to show the idea
def my_collate(batch):
    data = [item[0] for item in batch]
    target = [item[1] for item in batch]
    target = torch.LongTensor(target)
    return [data, target]


def show_image_batch(img_list, title=None):
    num = len(img_list)
    fig = plt.figure()
    for i in range(num):
        ax = fig.add_subplot(1, num, i+1)
        ax.imshow(img_list[i].numpy().transpose([1,2,0]))
        ax.set_title(title[i])

    plt.show()

#  do not do randomCrop to show that the custom collate_fn can handle images of different size
train_transforms = transforms.Compose([transforms.Scale(size = 224),
                                       transforms.ToTensor(),
                                       ])

# change root to valid dir in your system, see ImageFolder documentation for more info
train_dataset = datasets.ImageFolder(root="/hd1/jdhao/toyset",
                                     transform=train_transforms)

trainset = DataLoader(dataset=train_dataset,
                      batch_size=4,
                      shuffle=True,
                      collate_fn=my_collate, # use custom collate function here
                      pin_memory=True)

trainiter = iter(trainset)
imgs, labels = trainiter.next()

# print(type(imgs), type(labels))
show_image_batch(imgs, title=[train_dataset.classes[x] for x in labels])

'''