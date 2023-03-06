import os
import random
import string
from PIL import ImageFilter, Image, ImageOps
import numpy as np
import augly.image as imaugs
import augly.utils as utils
import augly.image.functional as F
import augly.image.utils as imutils
from typing import Any, Callable, Dict, List, Optional, Tuple, Union
import torchvision.transforms as transforms

# 随机选出RGB三颜色
def random_RGB(): return (random.randint(0, 255),
                          random.randint(0, 255), random.randint(0, 255))


# 随机添加的文字所组成的池子
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
string_pool = string.ascii_letters*2 + string.digits + ' '
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
string_pool_letter = string.ascii_letters + string.digits


def get_ramdom_string(length=15):
    # random.sample的用法，多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
    letter_list = [random.choice(string_pool_letter)] + random.sample(
        string_pool, length-2) + [random.choice(string_pool_letter)]
    random_str = ''.join(letter_list)
    return random_str

# MEME数据增强


class Meme(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        rgb = random_RGB()
        rgb_txt = random_RGB()
        text_len = random.randint(5, 9)
        text = get_ramdom_string(text_len)
        try:
            result_img = imaugs.meme_format(
                input_img,
                text=text,
                caption_height=random.randint(50, 200),
                meme_bg_color=rgb,
                text_color=rgb_txt,
            )
        except OSError:
            return input_img
        return result_img

# 打乱图像中的像素

class ShufflePixels(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        fact = random.randint(2, 8) * 0.1
        result_img = imaugs.shuffle_pixels(input_img, factor=fact)

        return result_img


# 随机像素化
class PixelizationRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        ratio = random.uniform(0.1, 0.5)
        result_img = imaugs.pixelization(input_img, ratio=ratio)

        return result_img

# 随机改变图像的亮度
class BrightnessRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        factor = random.uniform(0.1, 2.0)
        result_img = imaugs.brightness(input_img, factor=factor)

        return result_img

# 随机改变图像的饱和度
class SaturationRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        factor = random.randint(1, 50) * 0.1
        result_img = imaugs.saturation(input_img, factor=factor)

        return result_img

# 随机将图像改为灰度图


class GrayscaleRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        gray_mode = random.choice(["luminosity", "average"])
        result_img = imaugs.grayscale(input_img, mode=gray_mode)

        return result_img

# 随机模糊


class BlurRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        factor = random.uniform(2, 10)
        result_img = imaugs.blur(input_img, radius=factor)

        return result_img

# 随机增强图像对比度


class SharpenRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        factor = random.randint(5, 30)
        result_img = imaugs.sharpen(input_img, factor=factor)

        return result_img

# 随机改变图像质量


class JPEGEncodeAttackRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        q = random.randint(5, 20)
        result_img = imaugs.encoding_quality(input_img, quality=q)

        return result_img

# 随机图像滤波器
class FilterRandom(object):
    def __init__(self):
        self.filter_list = [
            ImageFilter.UnsharpMask,
            ImageFilter.EDGE_ENHANCE,
            ImageFilter.EDGE_ENHANCE_MORE,
            ImageFilter.SMOOTH_MORE,
            ImageFilter.MaxFilter(5),
            ImageFilter.MinFilter(5),
            # 提取线稿
            ImageFilter.CONTOUR,
            # 浮雕效果
            ImageFilter.EMBOSS,
            # 查找边缘（图像绝大部分会变成黑色）
            ImageFilter.FIND_EDGES,
        ]

    def __call__(self, input_img):
        f = random.choice(self.filter_list)
        result_img = imaugs.apply_pil_filter(input_img, filter_type=f)

        return result_img

# 随机视角转换
class PerspectiveTransform(object):
    def __init__(self):
        pass
    def __call__(self, input_img):
        sig = random.randint(10, 20) * 10.0
        dx = random.randint(1, 10)*0.1
        dy = random.randint(1, 10)*0.1
        seed = random.randint(1, 100)
        aug = imaugs.perspective_transform(input_img, sigma=sig, dx=dx, dy=dy, seed=seed)
        result_img = aug(input_img)

        return result_img


# 随机叠加文字
class OverlayTextRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):

        text = []
        text_list = range(1000)
        width = random.randint(5, 10)
        # 随机添加1-3次 字符串
        for _ in range(random.randint(1, 3)):
            text.append(random.sample(text_list, width))

        aug = imaugs.OverlayText(
            text=text,
            opacity=random.uniform(0.5, 1.0),
            font_size=random.uniform(0.1, 0.4),
            color=random_RGB(),
            x_pos=random.randint(0, 60) * 0.01,
            y_pos=random.randint(0, 60) * 0.01,
        )

        result_img = aug(input_img)

        return result_img

# 随机添加padding
class PadRandom(object):
    def __init__(self):
        pass

    def __call__(self, input_img):
        color = random_RGB()
        result_img = None
        if random.uniform(0, 1) > 0.5:
            # 将原图像pad成正方形
            result_img = imaugs.pad_square(
                input_img,
                color=color,
            )
        else:
            # 直接加一个边框
            result_img = imaugs.pad(
                input_img,
                color=color,
                w_factor=random.uniform(0.1, 0.3),
                h_factor=random.uniform(0.1, 0.3),
            )

        return result_img

# 垂直水平翻转裁减
class VerticalHorionalConvert(object):
    def __init__(self):
        self.wh_ratio = 9/16.
        # 1280x720, 1138x640, 1024x576, 960x540,

    def __call__(self, img):
        w, h = img.size
        if w < h:  # vertical -> horional
            new_h = w * random.uniform(0.5, 1)
            h_ratio = new_h/h
            y1 = (1-h_ratio)/2
            y2 = y1 + h_ratio
            result_img = imaugs.crop(img, x1=0, x2=1, y1=y1, y2=y2)

        else:  # horional -> vertical
            new_w = h * random.uniform(0.5, 1)
            w_ratio = new_w/w
            x1 = (1-w_ratio)/2
            x2 = x1 + w_ratio
            result_img = imaugs.crop(img, x1=x1, x2=x2, y1=0, y2=1)

        return result_img

# 制作抖音风格的图片
class DouyinFilter(object):
    def __init__(self):
        pass

    def __call__(self, input_img):

        array_orig = np.array(input_img)

        if array_orig.shape[0] <= 20 or array_orig.shape[1] <= 40:
            # print('[DouyinFilter]', array_orig.shape)
            return input_img

        array_r = np.copy(array_orig)
        array_r[:, :, 1:3] = 255  # cyan

        array_b = np.copy(array_orig)
        array_b[:, :, 0:2] = 255  # Y

        array_g = np.copy(array_orig)
        array_g[:, :, [0, 2]] = 255  # R

        result_array = array_r[:-20, 40:, :] + \
            array_b[20:, :-40, :] + array_g[10:-10, 20:-20, :]
        result_array[result_array > 255] = 255

        result = Image.fromarray(result_array)

        return result


# 将图片粘贴到一个随机背景上
class OverlayImage():
    def __init__(self) -> None:
        self.opacity: float = random.uniform(0.6, 1.0)
        # 叠加图像的比率
        self.overlay_size: float = random.uniform(0.2, 0.7)
        self.x_pos: float =random.uniform(0.3, 0.7)
        self.y_pos: float = random.uniform(0.3, 0.7)

    def __call__(self, overlay):
        bkgimg_base_path = '/data/sswang/data/isc_data/subset/training_data_background/'
        bkgimg_list = os.listdir(bkgimg_base_path)
        background_image = os.path.join(bkgimg_base_path, random.choice(bkgimg_list))
        result_img = imaugs.overlay_onto_background_image(background_image = background_image, 
                                                            image = overlay,
                                                            opacity=self.opacity,
                                                            overlay_size=self.overlay_size,
                                                            x_pos=self.x_pos,
                                                            y_pos=self.y_pos,
                                                            scale_bg=False)
        return result_img

# 随机挑选1-2张emoji 粘贴到指定图像上
class OverlayEmojiRandom(object):
    def __init__(self):
        self.base_path = '/data/sswang/image_matching/data_augmentations/'
        self.emoji_path = os.path.join(self.base_path, "emojis")
        self.emoji_list = os.listdir(self.emoji_path)

    def __call__(self, input_img):
        result_img = input_img
        for _ in range(random.randint(1, 2)):
            emoji = os.path.join(self.emoji_path, random.choice(self.emoji_list))
            result_img = imaugs.overlay_emoji(result_img,
                                                emoji_path=emoji,
                                                opacity=random.uniform(0.3, 1.0),
                                                emoji_size=random.uniform(0.2, 0.5),
                                                x_pos=random.randint(0, 50)*0.01,
                                                y_pos=random.randint(0, 50)*0.01,)
        return result_img


# 随机增加条纹线
class OverlayStripes():
    def __init__(self) -> None:
        self.line_width: float = random.uniform(0.2, 0.5)
        self.line_color: float = random_RGB()
        self.line_angle: float = random.randint(-360, 360)
        self.line_density: float = random.uniform(0.3, 0.8)
        self.line_type = random.choice(['dotted', 'dashed', 'solid'])
        self.line_opacity: float = random.uniform(0.5, 1.0)

    def __call__(self, input_img):
        result_img = imaugs.overlay_stripes(input_img, 
                                                   line_width=self.line_width,
                                                   line_color=self.line_color, 
                                                    line_angle=self.line_angle,
                                                    line_density=self.line_density,
                                                    line_type=self.line_type,
                                                    line_opacity=self.line_opacity)
        return result_img

# 随机裁减并生成指定尺寸大小
class RandomCropResize():
    def __init__(self) -> None:
        self.x1 = random.triangular(0.3, 0.49)
        self.y1 = random.triangular(0.3, 0.49)
        self.x2 = random.triangular(self.x1, 0.9)
        self.y2 = random.triangular(self.y1, 0.9)

    def __call__(self, input_img, width=224, height=224):
        result_img = imaugs.crop(input_img,
                                x1 = self.x1,
                                y1 = self.y1,
                                x2 = self.x2,
                                y2 = self.y2)
        result_img = imaugs.resize(result_img, width=width, height=height)
        return result_img


class OverlayOntoScreenshot():
    def __init__(self) -> None:
        pass

    def __call__(self, input_img):
        result_img = imaugs.overlay_onto_screenshot(input_img,
                                                    max_image_size_pixels=None,
                                                    crop_src_to_fit=True,
                                                    resize_src_to_match_template=True)
        return result_img

class ToRGB:
    # 将图片的颜色模式转为RGB
    def __call__(self, x):
        return x.convert("RGB")


class Solarization(object):
    # 将高于阈值的所有像素值反转，阈值仅表示图像分割
    # 用法： PIL.ImageOps.solarize(image, threshold=130)
    def __call__(self, x):
        return ImageOps.solarize(x)



class TorchvisionTrans(object):
    def __init__(self):
        self.trans_compose = [[
            ToRGB(),
            transforms.RandomResizedCrop(224, scale=(0.08, 1.0)),
            transforms.RandomApply([transforms.ColorJitter(0.4, 0.4, 0.2, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomHorizontalFlip(),
            transforms.RandomApply([Solarization()], p=0.2),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ]
        ]
    def __call__(self, input_img):
        transform = random.choice(self.trans_compose)
        result_img = transform(input_img)

        return result_img

# augly_trans_list = [
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom()],
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom()],
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom()],
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom()],
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom(),PadRandom(),OverlayEmojiRandom(),OverlayImage(),DouyinFilter()],
#     [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom(),PadRandom(),OverlayEmojiRandom(),OverlayImage(),OverlayOntoScreenshot(),TorchvisionTrans(), DouyinFilter(), FilterRandom()]
# ]

class Augly_Trans_List():
    def __init__(self) -> None:
        self.augly_trans_list = [
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom()],
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(), OverlayEmojiRandom()],
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),OverlayEmojiRandom()],
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom(),OverlayEmojiRandom()],
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom(),PadRandom(),OverlayEmojiRandom(),OverlayImage(),DouyinFilter()],
    [BrightnessRandom(),SharpenRandom(),SaturationRandom(),BlurRandom(), Meme(), GrayscaleRandom(),PixelizationRandom(),VerticalHorionalConvert(),OverlayTextRandom(),RandomCropResize(),ShufflePixels(),JPEGEncodeAttackRandom(),PadRandom(),OverlayEmojiRandom(),OverlayImage(),OverlayOntoScreenshot(), DouyinFilter(), FilterRandom()]
]