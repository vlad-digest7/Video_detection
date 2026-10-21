from ultralytics import YOLO

model = YOLO("yolov8n.pt")

res = model.train(
    data = 'D:\\Games\\project\\Final_Car_Accident_DataSet.v1-yolo8.yolov8\\data.yaml',
    epochs = 100,
    imgsz = 640,
    batch = 16,
    device = 'cpu'
)

