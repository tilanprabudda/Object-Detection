from ultralytics import YOLO
import cv2

model = YOLO ('../yolo-weight/yolov8l.pt')
results = model ("images/7.jpg", show = True)
cv2.waitKey(0)
