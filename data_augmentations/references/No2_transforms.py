import random
from PIL import ImageFilter, ImageOps
import torchvision.transforms as transforms


class ToRGB:
    # 将图片的颜色模式转为RGB
    def __call__(self, x):
        return x.convert("RGB")


class Solarization(object):
    # 将高于阈值的所有像素值反转，阈值仅表示图像分割
    # 用法： PIL.ImageOps.solarize(image, threshold=130)
    def __call__(self, x):
        return ImageOps.solarize(x)


class GaussianBlur(object):
    """
        高斯模糊
        Gaussian blur augmentation in SimCLR https://arxiv.org/abs/2002.05709
    """

    def __init__(self, sigma=[0.1, 2.0]):
        self.sigma = sigma

    def __call__(self, x):
        # PIL.ImageFilter.GaussianBlur()方法创建高斯模糊滤镜。
        # 用法： PIL.ImageFilter.GaussianBlur(radius=5) 
        # radius-模糊半径。通过改变半径的值，得到不同强度的高斯模糊图像。这里取一个0.1-2之间的随机数
        sigma = random.uniform(self.sigma[0], self.sigma[1])
        x = x.filter(ImageFilter.GaussianBlur(radius=sigma))
        return x

"""
transforms.ColorJitter(brightness, contrast, saturation, hue)

"""
def byol_transform():
    transform_q = transforms.Compose(
        [
            ToRGB(),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=1.0),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )

    transform_k = transforms.Compose(
        [
            ToRGB(),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomApply([GaussianBlur([0.1, 2.0])], p=0.1),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
    )
    return [transform_q, transform_k]


def typical_imagenet_transform(train):
    if train:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.RandomResizedCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    else:
        transform = transforms.Compose(
            [
                ToRGB(),
                transforms.Resize(256),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
            ]
        )
    return transform



