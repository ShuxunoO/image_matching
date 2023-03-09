# image_matching
Some Image Matching Algorithms for Learning

Win10 训练CLI命令
> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

> yolo predict model=C:/Users/Lenovo/Desktop/AI/image_matching/preprocessing/runs/detect/train/weights/best.pt source=C:/Users/Lenovo/Desktop/AI/image_matching/data/news_BA/

> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

155服务器
### 训练
> yolo task=detect mode=train model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train6/weights/best.pt data=/datassd2/sswang/image_matching/data/isc_data/yolo_training/overlay_detect_data.yaml epochs=10 imgsz=512 workers=2 device=0,1,2,3 resume=True

### 预测
> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train6/weights/best.pt source=/datassd2/sswang/image_matching/data/news_BA save=True conf=0.5

> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train6/weights/best.pt source=/datassd2/sswang/image_matching/data/isc_data/query_img2_subset device=1,2,3,4 save=True conf=0.3 save_crop=True