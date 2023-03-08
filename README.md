# image_matching
Some Image Matching Algorithms for Learning

训练CLI命令
> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2

> yolo predict model=C:/Users/Lenovo/Desktop/AI/image_matching/preprocessing/runs/detect/train/weights/best.pt source=C:/Users/Lenovo/Desktop/AI/image_matching/data/news_BA/

> yolo task=detect model=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/yolov8l.yaml data=C:/Users/Lenovo/Desktop/AI/image_matching/data/yolo_training_10K/overlay_detect_data.yaml epochs=1 imgsz=224 workers=2