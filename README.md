# image_matching
Some Image Matching Algorithms for Learning

Win10 训练CLI命令
> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

> yolo predict model=C:/Users/Lenovo/Desktop/AI/image_matching/preprocessing/runs/detect/train/weights/best.pt source=C:/Users/Lenovo/Desktop/AI/image_matching/data/news_BA/

> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

155服务器
### 训练
> yolo task=detect mode=train model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train8/weights/best.pt data=/datassd2/sswang/image_matching/data/isc_data/yolo_training/overlay_detect_data.yaml epochs=20 imgsz=512 workers=8 device=0,1 batch=50 resume=True


### 断点回复训练
> yolo task=detect mode=train model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train6/weights/best.pt data=/datassd2/sswang/image_matching/data/isc_data/yolo_training/overlay_detect_data.yaml epochs=20 imgsz=512 workers=8 device=0,1,5,6,7 batch=20 resume=True

### 预测
> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train8/weights/best.pt source=/datassd2/sswang/image_matching/data/news_BA device=0,1,2 save=True conf=0.5

> yolo predict model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train8/weights/best.pt source=/datassd2/sswang/image_matching/data/isc_data/query_img2_subset device=0,1,2,3,4 save=True conf=0.5

### 测试
边缘腐蚀
> yolo detect val data=/datassd2/sswang/image_matching/data/isc_data/test_subset/edge_corrosion/overlay_detect_data.yaml model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train8/weights/best.pt  device=0

九宫格
> yolo detect val data=/datassd2/sswang/image_matching/data/isc_data/test_subset/gride_concate/overlay_detect_data.yaml model=/datassd2/sswang/image_matching/preprocessing/runs/detect/train8/weights/best.pt device=0
