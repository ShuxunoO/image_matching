import sys
sys.path.append('..')





print(asset_map.bath_path)


# import random

# # from data_augment_augly import *

# # 随机裁减并生成指定尺寸大小
# class RandomCropResize():
#     def __init__(self) -> None:
#         self.x1 = random.uniform(0.2, 0.49)
#         self.y1 = random.uniform(0.2, 0.49)
#         self.x2 = random.uniform(0.5, 0.8)
#         self.y2 = random.uniform(0.5, 0.8)

#     def __call__(self, input_img, width, height):
#         result_img = imaugs.crop(input_img,
#                                 x1 = self.x1,
#                                 y1 = self.y1,
#                                 x2 = self.x2,
#                                 y2 = self.y2)
#         result_img = imaugs.resize(result_img, width=width, height=height)
#         return result_img

# augmentor = RandomCropResize()
# print(augmentor.x1)
# type(augmentor.x1)