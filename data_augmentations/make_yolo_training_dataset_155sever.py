import os
import random
from PIL import Image
from tqdm import tqdm
from utils.data_augmentation_for_yolo_training import generate_overlay_aug, generate_bg_aug, final_aug_, overlay_image


# 155服务器资源
trainingdata_path = '/datassd2/sswang/image_matching/data/isc_data/training_imgs/training/'
face_img_path = '/datassd2/sswang/image_matching/data/isc_data/training_imgs/faces/'
output_base_path = '/datassd2/sswang/image_matching/data/isc_data/yolo_training/'

# 检查路径是否存在，不存在就创建一个文件夹
def check_dir(dir_path):
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)

# 保存增强之后的图片和yolo标签
def save_auged_file(out_path, folder_name, img_name, yolo_label, final_img):
    base_path = os.path.join(out_path, folder_name)
    output_img_path = os.path.join(base_path, 'images')
    output_label_path = os.path.join(base_path, 'labels')

    # 检查文件夹
    check_dir(output_img_path)
    check_dir(output_label_path)

    # 创建标签文件,存储标签信息
    label_name = 'Aug_' + img_name.split('.')[0] + '.txt'
    yolo_label_path = os.path.join(output_label_path, label_name)
    with open(yolo_label_path, 'w') as f:
        str_ = str(yolo_label[0])
        for item in yolo_label[1:]:
            str_ = str_ + " " + str(item)
        f.write(str_)
        f.close()
    # print("{} saved".format(label_name))

    # 存储增强之后的图片
    aug_image_name = 'Aug_' + img_name
    final_img_path = os.path.join(output_img_path, aug_image_name)
    final_img.save(final_img_path)
    # print("{} saved".format(aug_image_name))

# 读取所有训练图片的名称
img_list = os.listdir(trainingdata_path)
bg_img_list = img_list[:19999]
face_img_list = os.listdir(face_img_path)
overlay_list = tqdm(face_img_list[:10000], desc='Augmenting images',unit = 'image')


# 按照训练集：测试集： 验证集 = 7：2: 1 的比例构造数据集
counter = 0
aug_level = 3
for img in overlay_list:
    # 每增强十轮之后重新构造数据增强对象
    if counter % 10 == 0:
        bg_aug = generate_bg_aug(width=512, height=512, aug_level = aug_level)
        overlay_aug = generate_overlay_aug(aug_level = aug_level)
        final_aug = final_aug_(aug_level = aug_level)

    # 加载图片
    bg_img_name = random.choice(bg_img_list)
    bg_img = Image.open(os.path.join(trainingdata_path, bg_img_name))
    overlay_img = Image.open(os.path.join(face_img_path, img))

    # 对背景图片数据增强
    bg_img = bg_aug(bg_img)

    # 对贴画图片进行数据增强
    overlay_img = overlay_aug(overlay_img)

    # 将贴图 贴在背景图上
    overlaied_img, yolo_label = overlay_image(
        bg_img=bg_img, overlay=overlay_img)

    # 对最后生成的图片进行数据增强
    final_img = final_aug(overlaied_img)

    # 存储增强之后的图片和标签
    # 0--6999编号的图片被放在train文件夹中
    if 0 <= counter < 6999:
        save_auged_file(output_base_path, folder_name='train',
                        img_name=img, yolo_label=yolo_label, final_img=final_img)

    # 7000--8999编号的图片被放在test文件夹中
    elif 7000 <= counter < 8999:
        save_auged_file(output_base_path, folder_name='test',
                        img_name=img, yolo_label=yolo_label, final_img=final_img)

    # 9000--9999编号的图片放在valid文件夹中
    else:
        save_auged_file(output_base_path, folder_name='valid',
                        img_name=img, yolo_label=yolo_label, final_img=final_img)

    counter += 1
    # if counter == 100:
    #     break
