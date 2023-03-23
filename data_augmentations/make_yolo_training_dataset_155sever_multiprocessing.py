import os
import random
import time
import multiprocessing as mp
from PIL import Image
from tqdm import tqdm
from utils.data_augmentation_for_yolo_training import generate_overlay_aug, generate_bg_aug, final_aug_, overlay_image, concat_images


# 155服务器资源
# 加载贴图资源
overlay_img_path = '/datassd2/sswang/image_matching/data/isc_data/training_imgs/training/'
overlay_img_list = os.listdir(overlay_img_path)

# 加载背景图片资源
bg_img_path = '/datassd2/sswang/image_matching/data/isc_data/training_imgs/training_bg/'
bg_img_list = os.listdir(bg_img_path)

# 加载人脸图片资源
face_img_path = '/datassd2/sswang/image_matching/data/isc_data/training_imgs/faces/'
face_img_list = os.listdir(face_img_path)

output_base_path = '/datassd2/sswang/image_matching/data/isc_data/yolo_training/'
# output_base_path = '/datassd2/sswang/image_matching/data/test_output/'



def check_dir(dir_path):
    """
        检查路径是否存在，不存在就创建一个文件夹

    Args:
        dir_path (str): 待检查的路径
    """
    if not os.path.exists(dir_path):
        os.makedirs(dir_path)


def random_choice_bgimg(bg_img_list):
    """
        随机选择一个背景图片

    Args:
        bg_img_list (list): 背景图片名称列表

    Returns:
        PIL.IMG: 返回打开的背景图片
    """
    bg_img_name = random.choice(bg_img_list)
    return Image.open(os.path.join(bg_img_path, bg_img_name))


def save_auged_file(out_path, folder_name, img_name, yolo_label, final_img):
    """
        保存增强之后的图片和yolo标签

    Args:
        out_path (str): 输出文件的基础路径
        folder_name (str): 用于保存文件夹名称
        img_name (str): 图片的名称
        yolo_label (obj): YOLO label对象
        final_img (obj): 最后生成的图片对象
    """
    base_path = os.path.join(out_path, folder_name)
    output_img_path = os.path.join(base_path, 'images')
    output_label_path = os.path.join(base_path, 'labels')

    # 检查文件夹
    check_dir(output_img_path)
    check_dir(output_label_path)

    # 创建标签文件,存储标签信息
    label_name = 'Aug_New' + img_name.split('.')[0] + '.txt'
    yolo_label_path = os.path.join(output_label_path, label_name)
    with open(yolo_label_path, 'w') as f:
        str_ = str(yolo_label[0])
        for item in yolo_label[1:]:
            str_ = str_ + " " + str(item)
        f.write(str_)
        f.close()
    print("{} saved".format(label_name))

    # 存储增强之后的图片
    aug_image_name = 'Aug_New' + img_name
    final_img_path = os.path.join(output_img_path, aug_image_name)
    final_img.convert('RGB').save(final_img_path)
    print("{} saved".format(aug_image_name))


# 数据增强函数
def data_augmentation(overlay_list, aug_level=0, folder_name='train'):
    """
        对用于贴图检测的数据集进行数据增强，生成yolo训练数据集
        增强的方式包括：对背景图片进行数据增强，对贴图进行数据增强，对最后生成的图片进行数据增强

    Args:

        overlay_list (list): 用于数据增强的贴图地址列表
        aug_level (int, optional): 数据增强的等级. Defaults to 0.
        folder_name (str, optional): 生成的数据集存储的文件夹名称. Defaults to 'train'.

    """

    counter = 0
    for img in overlay_list:
        # 每增强十轮之后重新构造数据增强对象
        if counter % 50 == 0:
            bg_aug = generate_bg_aug(
                width=512, height=512, aug_level=aug_level)
            overlay_aug = generate_overlay_aug(aug_level=aug_level)
            final_aug = final_aug_(aug_level=aug_level)

        # 加载图片
        bg_img = random_choice_bgimg(bg_img_list)
        overlay_img = Image.open(os.path.join(overlay_img_path, img)).convert('RGBA')

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

        save_auged_file(output_base_path, folder_name=folder_name,
                        img_name=img, yolo_label=yolo_label, final_img=final_img)
        counter += 1

# 进程测试函数
def test(p):
       print(p)
       time.sleep(0.1)

def chunk_list(original_list, size):
    """
        将列表等分成多个子列表

    Args:
        original_list (list): 待分割的列表
        size (int): 子列表长度

    Returns:
        _type_: 分割后包含等长度为size的子列表的二维列表
    """
    return [original_list[i:i+size] for i in range(0, len(original_list), size)]



if __name__ == "__main__":

    start_time = time.time()

    # 将待增强的训练集做数据切分，overlay_img_subset_1~2 用作训练集， overlay_img_subset_3 用验证集
    overlay_img_subset_1 = overlay_img_list[0:50000]
    overlay_img_subset_2 = overlay_img_list[50000:65000]
    overlay_img_subset_3 = overlay_img_list[65000:70000]
    face_img_subset_1 = face_img_list[0:6]
    face_img_subset_2 = face_img_list[24000:25800]

    # overlay_img_for_each_pool = chunk_list(overlay_img_subset_1, 10000)
    # for index in range(len(overlay_img_for_each_pool)):
    #     overlay_img_for_each_processing = chunk_list(overlay_img_for_each_pool[index], 100)
    #     # 创建一个包含20个进程的进程池
    #     pool = mp.Pool(processes = 20)
    #     for sub_list in overlay_img_for_each_processing:
    #         # 维持执行的进程总数为20，当一个进程执行完后启动一个新进程.
    #         pool.apply_async(data_augmentation, args=(sub_list, index + 1, "train",))
    #         # pool.apply_async(test, args=(item_list,))
    # pool.close()
    # pool.join()

    # overlay_img_for_each_processing = chunk_list(overlay_img_subset_2, 100)
    # # 创建一个包含20个进程的进程池
    # pool = mp.Pool(processes = 20)
    # for sub_list in overlay_img_for_each_processing:
    #     # 维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
    #     pool.apply_async(data_augmentation, args=(sub_list, 5, "train",))
    # pool.close()
    # pool.join()


    # overlay_img_for_each_pool_3 = chunk_list(overlay_img_subset_3, 1000)
    # for index in range(len(overlay_img_for_each_pool_3)):
    #     overlay_img_for_each_processing = chunk_list(overlay_img_for_each_pool_3[index], 100)
    #     # 创建一个包含20个进程的进程池
    #     pool = mp.Pool(processes = 10)
    #     for sub_list in overlay_img_for_each_processing:
    #         # 维持执行的进程总数为10，当一个进程执行完后启动一个新进程.
    #         pool.apply_async(data_augmentation, args=(sub_list, index + 1, "valid",))
    # pool.close()
    # pool.join()


    # face_img_for_each_pool = chunk_list(face_img_subset_1,1)
    # # 创建一个包含20个进程的进程池
    # pool = mp.Pool(processes = 10)
    # for index in range(len(face_img_for_each_pool)):
    #     overlay_img_for_each_processing = chunk_list(face_img_for_each_pool[index], 1)
    #     for sub_list in overlay_img_for_each_processing:
    #         # 维持执行的进程总数为20，当一个进程执行完后启动一个新进程.
    #         pool.apply_async(data_augmentation, args=(sub_list, index, "train",))
    # pool.close()
    # pool.join()




    concat_images(img_base_path, col, row, width, height):

    end_time = time.time()
    print("time cost: {}".format(end_time - start_time))