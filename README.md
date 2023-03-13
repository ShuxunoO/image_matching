# image_matching
Some Image Matching Algorithms for Learning

Win10 训练CLI命令
> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

> yolo predict model=C:/Users/Lenovo/Desktop/AI/image_matching/preprocessing/runs/detect/train/weights/best.pt source=C:/Users/Lenovo/Desktop/AI/image_matching/data/news_BA/

> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

155服务器
### 训练
> yolo task=detect mode=train model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train6/weights/best.pt data=/datassd2/sswang/image_matching/data/isc_data/yolo_training/overlay_detect_data.yaml epochs=50 imgsz=512 workers=2 device=0,1,2,3 resume=True

### 预测
> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train7/weights/best.pt source=/datassd2/sswang/image_matching/data/news_BA device=1,2,3,4 save=True conf=0.3 

> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train7/weights/best.pt source=/datassd2/sswang/image_matching/data/isc_data/query_img2_subset device=1,2,3,4 save=True conf=0.3 save_crop=True



Traceback (most recent call last):
  File "/datassd2/sswang/image_matching/data_augmentations/make_yolo_training_dataset_155sever.py", line 68, in <module>
    bg_img = bg_aug(bg_img)
  File "/home/sswang/anaconda3/envs/img_matching/lib/python3.9/site-packages/augly/image/composition.py", line 89, in __call__
    image = transform(
  File "/home/sswang/anaconda3/envs/img_matching/lib/python3.9/site-packages/augly/image/transforms.py", line 64, in __call__
    return self.apply_transform(image, metadata, bboxes, bbox_format)
  File "/home/sswang/anaconda3/envs/img_matching/lib/python3.9/site-packages/augly/image/transforms.py", line 1702, in apply_transform
    return F.pixelization(
  File "/home/sswang/anaconda3/envs/img_matching/lib/python3.9/site-packages/augly/image/functional.py", line 1900, in pixelization
    aug_image = image.resize((int(width * ratio), int(height * ratio)))
  File "/home/sswang/anaconda3/envs/img_matching/lib/python3.9/site-packages/PIL/Image.py", line 2192, in resize
    return self._new(self.im.resize(size, resample, box))
ValueError: height and width must be > 0