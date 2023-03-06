import os
import random
import string
import numpy as np
from PIL import ImageFilter, Image, ImageOps
import augly.image as imaugs
import augly.image.transforms as transaugs
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


# 随机取出一个长度介于[5,30]字符串
def get_ramdom_string():
    length=random.randint(5, 30)
    # random.sample的用法，多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
    letter_list = [random.choice(string_pool_letter)] + random.sample(
        string_pool, length-2) + [random.choice(string_pool_letter)]
    random_str = ''.join(letter_list)
    return random_str

# 随机取出一种图片过滤方法
def get_ramdom_imagefilter():
    filter_list = [
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
    return random.choice(filter_list)

# 随机取出一个emoji地址
def get_ramdom_emoji():
        emoji_path = 'C:/Users/Lenovo/Desktop/AI/yolo/data/emojis/'
        emoji_list = os.listdir(emoji_path)
        return os.path.join(emoji_path, random.choice(emoji_list))

# 随机取出一种虚线类型
def get_line_type():
    return random.choice(['dotted', 'dashed', 'solid'])

# 随机选出一个字体
def get_random_font():
        font_path = 'C:/Users/Lenovo/Desktop/AI/yolo/data/fonts'
        font_list = os.listdir(font_path)
        return os.path.join(font_path, random.choice(font_list))

# 返回随机选择的int型文字索引列表
def get_random_overlaytext():
    text_index_list = []
    for _ in range(random.randint(2, 3)):
        temp_list = []
        for i in range(random.randint(8, 15)):
            temp_list.append(random.randint(10, 5000))
        text_index_list.append(temp_list)
    return text_index_list

# 拿到一个随机的截屏背景
def get_random_screenshot_template():
        screenshot_path = 'C:/Users/Lenovo/Desktop/AI/yolo/data/screenshot_templates/'
        screenshot_list = [i for i in os.listdir(screenshot_path) if i.endswith('png')]
        return os.path.join(screenshot_path, random.choice(screenshot_list))



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

aug1 = imaugs.Compose(
    [
        # 改变图像明亮度
        imaugs.Brightness(factor=random.uniform(0, 3), p=0.6),

        # 改变图像对比度
        imaugs.Contrast(factor=random.uniform(0, 10),p=0.6),

        # 改变图片的锐度
        imaugs.Sharpen(factor=random.uniform(10,20),  p=0.6),

        # 随机裁减下来图像
        imaugs.Crop(x1=random.triangular(0.3, 0.49),
                    y1=random.triangular(0.3, 0.49),
                    x2=random.triangular(0.5, 0.9),
                    y2=random.triangular(0.5, 0.9), p=0.6),

        # 改变图像画质的压缩质量
        imaugs.EncodingQuality(quality=random.randint(0,100), p=1.0),

        # 将图像变为灰度图
        imaugs.Grayscale(mode=random.choice(['luminosity' ,'average']), p=0.6),

        # 图像过滤器
        imaugs.ApplyPILFilter(filter_type = get_ramdom_imagefilter(), p=0.6),

        # 水平翻转
        imaugs.HFlip(p=0.6),

        # 模因格式（在图片上方加文字）
        imaugs.MemeFormat(text=get_ramdom_string(), opacity=random.uniform(0.3,1), text_color=random_RGB(), caption_height=random.randint(20,200), meme_bg_color=random_RGB(),p=0.6),

        # 改变图像的透明度
        imaugs.Opacity(level=random.random(),p=0.6),

        # 在图像上叠加emoji
        imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5,1), emoji_size=random.uniform(0,1), x_pos=random.uniform(0.1,0.7), y_pos=random.uniform(0.3,0.7),p=1.0),
        imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5,1), emoji_size=random.uniform(0,1), x_pos=random.uniform(0.1,0.5), y_pos=random.uniform(0.3,0.7),p=0.6),
        imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.2,1), emoji_size=random.uniform(0,1), x_pos=random.uniform(0.5,0.7), y_pos=random.uniform(0.3,0.7),p=0.3),


        # 在图像上叠加线段
        imaugs.OverlayStripes(line_width=random.uniform(0.3,1), line_color=random_RGB(), line_angle=random.uniform(-180, 180), line_density=random.uniform(0.3,1), line_type=get_line_type(), line_opacity=random.uniform(0.5,1),p=0.6),

        # 在图片上叠加文字
        imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1,0.3), opacity=random.uniform(0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1,0.6), y_pos=random.uniform(0.1,0.7), p=0.6),

        # 把图片做成截图的样式
        imaugs.OverlayOntoScreenshot(template_filepath=get_random_screenshot_template(), template_bboxes_filepath='C:/Users/Lenovo/Desktop/AI/yolo/data/screenshot_templates/bboxes.json', max_image_size_pixels=None, crop_src_to_fit=False, resize_src_to_match_template=True,p=0.6),


        # 给图片加上边框
        imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(0.1, 0.3), color=random_RGB(), p=0.6),

        # 将图片pad成正方形
        imaugs.PadSquare(color=random_RGB(), p=0.6),

        # 改变图片的观察角度
        imaugs.PerspectiveTransform(sigma=random.randint(50,150), dx=random.uniform(0,10), dy=random.uniform(0,10), seed=random.randint(1,200), p=0.6),

        # 将图片像素化
        imaugs.Pixelization(ratio=random.random(),  p=0.6),

        # 改变图片的长宽比
        imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.6),

        # 给图片加上噪声（这个运算很慢）
        imaugs.RandomNoise(mean=random.random(), var=random.uniform(0.5, 1), seed=random.randint(10, 50),  p=0.6),

        # 图片随机像素化
        imaugs.RandomPixelization(min_ratio=0.1, max_ratio=1.0, p=0.6),

        # 图片旋转
        imaugs.Rotate(degrees=random.uniform(-180, 180), p=0.6),

        # 改变图像饱和度
        imaugs.Saturation(factor=random.random(), p=0.6),

        # 缩放图片
        imaugs.Scale(factor=random.random(), interpolation=None, p=0.6),

        # 打乱图片的像素分布（很费时间）
        imaugs.ShufflePixels(factor=random.uniform(0.3,1), seed=random.randint(10,100), p=0.6),

        #让图片变倾斜
        imaugs.Skew(skew_factor=random.uniform(-2, 2), axis=random.choice([0, 1]),  p=0.6),

        # 图片垂直翻转
        imaugs.VFlip(p=0.6),

        # 改变图像的长宽到指定尺寸
        imaugs.Resize(width=224, height=224, p=1.0)

        ]
            )