import os
import random
import string
import numpy as np
from PIL import ImageFilter, Image, ImageDraw
import augly.image as imaugs
from augly.image import utils as imutils
import augly.image.transforms as transaugs
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

# 随机选出RGB三颜色
def random_RGB():
    return (random.randint(0, 255), random.randint(0, 255), random.randint(0, 255))


# 随机添加的文字所组成的池子
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
string_pool = string.ascii_letters * 2 + string.digits + ' '
# abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789
string_pool_letter = string.ascii_letters + string.digits


# 随机取出一个长度介于[5,30]字符串
def get_ramdom_string():
    length = random.randint(5, 30)
    # random.sample的用法，多用于截取列表的指定长度的随机数，但是不会改变列表本身的排序
    letter_list = [random.choice(string_pool_letter)] + random.sample(
        string_pool, length - 2) + [random.choice(string_pool_letter)]
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
    # 155服务器上的emoji地址
    emoji_path = '/datassd2/sswang/image_matching/data_augmentations/emojis'

    # 116服务器上的emoji地址
    # emoji_path = '/datassd2/sswang/image_matching/data_augmentations/emojis/'

    # 本地的emoji地址
    # emoji_path = 'C:/Users/Lenovo/Desktop/AI/image_matching/data_augmentations/emojis/'
    emoji_list = os.listdir(emoji_path)
    return os.path.join(emoji_path, random.choice(emoji_list))


# 随机取出一种虚线类型
def get_line_type():
    return random.choice(['dotted', 'dashed', 'solid'])


# 随机选出一个字体
def get_random_font():
    font_path = ''
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
    screenshot_path = '/datassd2/sswang/image_matching/data_augmentations/screenshot_templates/'

    # 本地资源
    # screenshot_path = 'C:/Users/Lenovo/Desktop/AI/image_matching/data/screenshot_templates/'
    screenshot_list = [
        i for i in os.listdir(screenshot_path) if i.endswith('png')
    ]
    return os.path.join(screenshot_path, random.choice(screenshot_list))

# 随机选出一个mask
def get_random_mask():
    # 155服务器上的资源
    mask_path = '/datassd2/sswang/image_matching/data/masks'

    # 本地资源
    # screenshot_path = 'C:/Users/Lenovo/Desktop/AI/image_matching/data/screenshot_templates/'
    screenshot_list = os.listdir(mask_path)
    return os.path.join(mask_path, random.choice(screenshot_list))


# 将YOLO标签画出来
def draw_yolo_rectangle(yolo_label, img):

    _, x, y, width, height = yolo_label
    bg_width, bg_height = img.size

    # 将YOLO标签转化为ImageDraw.Draw.rectangle()方法支持的输入格式
    over_width = bg_width * width
    over_height = bg_height * height
    point1_xpos = bg_width * x - over_width / 2
    point1_ypos = bg_height * y - over_height / 2
    point2_xpos = point1_xpos + over_width
    point2_ypos = point1_ypos + over_height

    draw = ImageDraw.Draw(img)
    draw.rectangle([(point1_xpos, point1_ypos), (point2_xpos,
                    point2_ypos)], fill=None, outline="red", width=2)
    return img

# 将贴图粘贴到背景图上
def overlay_image(bg_img, overlay):
    """
        将overlay图层粘贴到bg_img图层上，之后返回对应的yolo格式的标签

    """
    overlay_size = random.uniform(0.3, 0.5)
    x_pos = random.uniform(0.1, 0.5)
    y_pos = random.uniform(0.1, 0.5)

    """
        把 image 贴到 background 上面

        image： 贴图

        background_image ： 背景图片

        opacity (float)：贴图的透明度

        overlay_size (float): 贴图的高度与背景图高度的比值

        x_pos (float)： 贴图相对于背景图像宽度在x轴的位置

        y_pos (float)： 贴图相对于背景图像高度在y轴的位置

        scale_bg (bool)： if True, the background image will be scaled up or down so that overlay_size is respected; if False, the source image will be scaled instead


    """
    aug_img = imaugs.overlay_onto_background_image(image=overlay, background_image=bg_img,
                                                    opacity=random.uniform(0.3, 1), overlay_size=overlay_size, x_pos=x_pos, y_pos=y_pos)

    # 背景图的宽和高
    bg_width, bg_height = bg_img.size

    # 贴图原始的宽和高
    img_width, img_height = overlay.size

    # 贴图实际的高度
    over_height = bg_height * overlay_size

    # 贴图实际的宽度
    over_width = img_width / img_height * over_height

    # 计算YOLO标签
    # 取 min()操作的目的是为了避免有标签值超过1的情况
    x = min((x_pos * bg_width + over_width / 2) / bg_width, 1.0)
    y = min((y_pos * bg_height + over_height / 2) / bg_height, 1.0)
    width = min(over_width / bg_width, 1.0)
    height = overlay_size
    yolo_label = (0, x, y, width, height)

    return aug_img, yolo_label


def mask_img(
    image: Union[str, Image.Image],
    mask : Image.Image,
    output_path: Optional[str] = None,
    metadata: Optional[List[Dict[str, Any]]] = None,
    bboxes: Optional[List[Tuple]] = None,
    bbox_format: Optional[str] = None,) ->(Image.Image):
    """
    Applies a mask to an image. The mask must be a PIL Image of the same size as the image.

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param mask: the path to a mask or a variable of type PIL.Image.Image

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: PIL.Image， 蒙版覆盖后的图片

    """
    # print("mask_img, Img type1: ", type(image))
    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    #  创建一个空白的RGBA图层
    blank_layer = Image.new('RGBA', image.size, (0,0,0,0))

    # 修改蒙版尺寸以匹配图片
    mask = mask.resize(image.size,Image.BILINEAR)
    masked_img = Image.composite(blank_layer, image, mask)
    src_mode = masked_img.mode
    imutils.get_metadata(metadata=metadata, function_name="mask_img", **func_kwargs)
    return imutils.ret_and_save_image(masked_img, output_path, src_mode)


def douyin_style(
        image: Union[str, Image.Image],
        output_path: Optional[str] = None,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,) ->(Image.Image):
    """

    给图片加上抖音滤镜

    @param image: the path to an image or a variable of type PIL.Image.Image
        to be augmented

    @param mask: the path to a mask or a variable of type PIL.Image.Image

    @param output_path: the path in which the resulting image will be stored.
        If None, the resulting PIL Image will still be returned

    @param metadata: if set to be a list, metadata about the function execution
        including its name, the source & dest width, height, etc. will be appended
        to the inputted list. If set to None, no metadata will be appended or returned

    @param bboxes: a list of bounding boxes can be passed in here if desired. If
        provided, this list will be modified in place such that each bounding box is
        transformed according to this function

    @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
        specify `bbox_format` if `bboxes` is provided. Supported bbox_format values are
        "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

    @returns: PIL.Image， 蒙版覆盖后的图片

    """
    image = imutils.validate_and_load_image(image)
    func_kwargs = imutils.get_func_kwargs(metadata, locals())

    array_orig = np.array(image)

    if array_orig.shape[0] <= 20 or array_orig.shape[1] <= 40:
        # print('[DouyinFilter]', array_orig.shape)
        return image

    array_r = np.copy(array_orig)
    array_r[:, :, 1:3] = 255  # cyan

    array_b = np.copy(array_orig)
    array_b[:, :, 0:2] = 255  # Y

    array_g = np.copy(array_orig)
    array_g[:, :, [0, 2]] = 255  # R

    result_array = array_r[:-20, 40:, :] + \
        array_b[20:, :-40, :] + array_g[20:, 40:, :]
    image = Image.fromarray(result_array)

    src_mode = image.mode
    imutils.get_metadata(metadata=metadata, function_name="douyin_style", **func_kwargs)

    return imutils.ret_and_save_image(image, output_path, src_mode)


# 对图片应用蒙版
class Mask_IMG(transaugs.BaseTransform):
    """对图片应用蒙版"""

    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        对图片应用蒙版

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        # 生成蒙版
        mask = Image.open(get_random_mask()).convert("RGBA")
        return mask_img(image, mask, metadata, bboxes, bbox_format)



# 制作抖音风格的图片
class DouyinFilter(transaugs.BaseTransform):
    """将图片转换为抖音风格"""

    def __init__(self, p: float = 1.0):
        """
        @param p: the probability of the transform being applied; default value is 1.0
        """
        super().__init__(p)

    def apply_transform(
        self,
        image: Image.Image,
        metadata: Optional[List[Dict[str, Any]]] = None,
        bboxes: Optional[List[Tuple]] = None,
        bbox_format: Optional[str] = None,
    ) -> Image.Image:
        """
        对图片应用蒙版

        @param image: PIL Image to be augmented

        @param metadata: if set to be a list, metadata about the function execution
            including its name, the source & dest width, height, etc. will be appended to
            the inputted list. If set to None, no metadata will be appended or returned

        @param bboxes: a list of bounding boxes can be passed in here if desired. If
            provided, this list will be modified in place such that each bounding box is
            transformed according to this function

        @param bbox_format: signifies what bounding box format was used in `bboxes`. Must
            specify `bbox_format` if `bboxes` is provided. Supported bbox_format values
            are "pascal_voc", "pascal_voc_norm", "coco", and "yolo"

        @returns: Augmented PIL Image
        """
        return douyin_style(image, metadata, bboxes, bbox_format)





# 对贴图进行增强
def generate_overlay_aug(aug_level=0):
    # 模式0
    over_aug0 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 1), p=0.3),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 2), p=0.3),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.3),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.3),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.2, 0.49),
                y1=random.triangular(0.2, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.1),

            # 水平翻转
            imaugs.HFlip(p=0.3),

            # 图片旋转
            imaugs.Rotate(degrees=random.uniform(-180, 180), p=0.3),

        ])

    # 模式1
    over_aug1 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 1), p=0.3),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 2), p=0.3),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.3),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.3),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.1),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.3),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.4),

            # 图片水平翻转
            imaugs.HFlip(p=0.2),

            # 图片垂直翻转
            imaugs.VFlip(p=0.2)

        ]
    )

    # 模式2
    over_aug2 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 2), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 3), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.2, 0.49),
                y1=random.triangular(0.2, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.1),

            # 水平翻转
            imaugs.HFlip(p=0.4),

            # 图片旋转
            imaugs.Rotate(degrees=random.uniform(-180, 180), p=0.4),

            # 改变图像画质的压缩质量
            imaugs.EncodingQuality(quality=random.randint(20, 100), p=0.2),

            # 模因格式（在图片上方加文字）
            imaugs.MemeFormat(text=get_ramdom_string(), opacity=random.uniform(0.3, 1), text_color=random_RGB(
            ), caption_height=random.randint(20, 200), meme_bg_color=random_RGB(), p=0.1),

            # 倾斜图片
            imaugs.Skew(skew_factor=random.uniform(-2, 2),
                        axis=random.choice([0, 1]), p=0.1),
        ]
    )

    # 模式3
    over_aug3 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 2), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 3), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # # 随机裁减
            # imaugs.Crop(
            #     x1=random.triangular(0.3, 0.49),
            #     y1=random.triangular(0.3, 0.49),
            #     x2=random.triangular(0.5, 0.9),
            #     y2=random.triangular(0.5, 0.9),
            #     p=0.1),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.3),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.05),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 100), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

        ]
    )

    # 模式4
    over_aug4 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 3), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 5), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(10, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.1),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.1),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.05),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 150), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.3),

            # # 在图像上叠加emoji
            # imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
            #     0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.3),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.3,0.7), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.4),

            # 将图片像素化
            imaugs.Pixelization(ratio=random.random(),  p=0.1),

        ]
    )

    # 模式5
    over_aug5 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 3), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 5), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(10, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            #随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.2),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.1),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.05),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 150), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.8),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.3),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.3, 0.6), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.15),

            # 将图片像素化
            imaugs.Pixelization(ratio=random.random(),  p=0.1),

            # 在图片上叠加文字
            imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1, 0.3), opacity=random.uniform(
                0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1, 0.6), y_pos=random.uniform(0.1, 0.7), p=0.2),

            # 给图片加上噪声（这个运算很慢）
            imaugs.RandomNoise(mean=random.random(), var=random.uniform(
                0.5, 1), seed=random.randint(10, 50),  p=0.08),
        ]
    )

    aug_list = [over_aug0, over_aug1, over_aug2,
                over_aug3, over_aug4, over_aug5]
    return aug_list[aug_level]


# 对背景图片进行增强
def generate_bg_aug(width=224, height=224, aug_level=0):
    # 模式0
    bg_aug0 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 1), p=0.3),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 2), p=0.3),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.3),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.3),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.2, 0.49),
                y1=random.triangular(0.2, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.3),

            # 水平翻转
            imaugs.HFlip(p=0.6),

            # 图片旋转
            imaugs.Rotate(degrees=random.uniform(-180, 180), p=0.6),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ])

    # 模式1
    bg_aug1 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 1), p=0.3),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 2), p=0.3),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.3),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.3),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.3),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.4),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.4),

            # 图片水平翻转
            imaugs.HFlip(p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ]
    )

    bg_aug2 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 2), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 3), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.2, 0.49),
                y1=random.triangular(0.2, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.3),

            # 水平翻转
            imaugs.HFlip(p=0.4),

            # 图片旋转
            imaugs.Rotate(degrees=random.uniform(-180, 180), p=0.4),

            # 改变图像画质的压缩质量
            imaugs.EncodingQuality(quality=random.randint(50, 100), p=0.2),

            # 模因格式（在图片上方加文字）
            imaugs.MemeFormat(text=get_ramdom_string(), opacity=random.uniform(0.3, 1), text_color=random_RGB(
            ), caption_height=random.randint(20, 200), meme_bg_color=random_RGB(), p=0.2),

            # 倾斜图片
            imaugs.Skew(skew_factor=random.uniform(-2, 2),
                        axis=random.choice([0, 1]), p=0.1),

            # 将图片pad成正方形
            imaugs.PadSquare(color=random_RGB(), p=0.2),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ]
    )

    bg_aug3 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 2), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 3), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(5, 15), p=0.2),


            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.3),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.3),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.2),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.2),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 150), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

            # 倾斜图片
            imaugs.Skew(skew_factor=random.uniform(-2, 2),
                        axis=random.choice([0, 1]), p=0.1),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ]
    )

    bg_aug4 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 3), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 5), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(10, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.3),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.1),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.1),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 150), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.5),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.3, 0.6), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.15),

            # 将图片像素化
            imaugs.Pixelization(ratio=random.random(),  p=0.1),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ]
    )

    bg_aug5 = imaugs.Compose(
        [
            # 改变图片亮度
            imaugs.Brightness(factor=random.uniform(0, 3), p=0.2),

            # 改变图片对比度
            imaugs.Contrast(factor=random.uniform(0, 5), p=0.2),

            # 锐化图片
            imaugs.Sharpen(factor=random.uniform(10, 20), p=0.2),

            # 改变图像饱和度
            imaugs.Saturation(factor=random.random(), p=0.2),

            # 随机裁减
            imaugs.Crop(
                x1=random.triangular(0.3, 0.49),
                y1=random.triangular(0.3, 0.49),
                x2=random.triangular(0.5, 0.9),
                y2=random.triangular(0.5, 0.9),
                p=0.1),

            # 将图像变为灰度图
            imaugs.Grayscale(mode=random.choice(
                ['luminosity', 'average']), p=0.1),

            # 改变图片的长宽比
            imaugs.RandomAspectRatio(min_ratio=0.3, max_ratio=3.0, p=0.3),

            # 图片垂直翻转
            imaugs.VFlip(p=0.3),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 给图片加上边框
            imaugs.Pad(w_factor=random.uniform(0.1, 0.3), h_factor=random.uniform(
                0.1, 0.3), color=random_RGB(), p=0.1),

            # 改变图片的观察角度
            imaugs.PerspectiveTransform(sigma=random.randint(50, 150), dx=random.uniform(
                0, 10), dy=random.uniform(0, 10), seed=random.randint(1, 200), p=0.3),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.5),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.3, 0.6), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.15),

            # 将图片像素化
            imaugs.Pixelization(ratio=random.random(),  p=0.1),

            # 在图片上叠加文字
            imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1, 0.3), opacity=random.uniform(
                0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1, 0.6), y_pos=random.uniform(0.1, 0.7), p=0.3),

            # 给图片加上噪声（这个运算很慢）
            imaugs.RandomNoise(mean=random.random(), var=random.uniform(
                0.5, 1), seed=random.randint(10, 50),  p=0.08),

            # 最后调节一下图片尺寸
            imaugs.Resize(width=width, height=height, p=1.0)
        ]
    )

    aug_list = [bg_aug0, bg_aug1, bg_aug2, bg_aug3, bg_aug4, bg_aug5]
    return aug_list[aug_level]


# 对最后生成的图片进行增强
def final_aug_(aug_level=0):

    # 模式0
    final_aug0 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),
        ]
    )

    # 模式1
    final_aug1 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.1),

        ]
    )

    # 模式2
    final_aug2 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.1),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.2, 0.5), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.1),

        ]
    )

    # 模式3
    final_aug3 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.1),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.2, 0.5), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.1),

            # 在图片上叠加文字
            imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1, 0.3), opacity=random.uniform(
                0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1, 0.6), y_pos=random.uniform(0.1, 0.7), p=0.1),

        ]
    )

    # 模式4
    final_aug4 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.1),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.2, 0.5), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.1),

            # 在图片上叠加文字
            imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1, 0.3), opacity=random.uniform(
                0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1, 0.6), y_pos=random.uniform(0.1, 0.7), p=0.1),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),
        ]
    )

    # 模式5
    final_aug5 = imaugs.Compose(
        [
            # 图片随机模糊
            imaugs.Blur(radius=random.uniform(0, 3), p=0.1),

            # 在图像上叠加emoji
            imaugs.OverlayEmoji(emoji_path=get_ramdom_emoji(), opacity=random.uniform(0.5, 1), emoji_size=random.uniform(
                0, 1), x_pos=random.uniform(0.1, 0.7), y_pos=random.uniform(0.3, 0.7), p=0.1),

            # 在图像上叠加线段
            imaugs.OverlayStripes(line_width=random.uniform(0.2, 0.5), line_color=random_RGB(), line_angle=random.uniform(
                -180, 180), line_density=random.uniform(0.3, 1), line_type=get_line_type(), line_opacity=random.uniform(0.5, 1), p=0.1),

            # 在图片上叠加文字
            imaugs.OverlayText(text=get_random_overlaytext(), font_size=random.uniform(0.1, 0.3), opacity=random.uniform(
                0.3, 1), color=random_RGB(), x_pos=random.uniform(0.1, 0.6), y_pos=random.uniform(0.1, 0.7), p=0.1),

            # 图像过滤器
            imaugs.ApplyPILFilter(filter_type=get_ramdom_imagefilter(), p=0.1),

            # 图片随机像素化
            imaugs.RandomPixelization(min_ratio=0.1, max_ratio=1.0, p=0.1),

        ]
    )
    aug_list_ = [final_aug0, final_aug1, final_aug2,
                final_aug3, final_aug4, final_aug5]
    # aug_list_ = [0, 1, 2, 3, 4, 5]
    return aug_list_[aug_level]
