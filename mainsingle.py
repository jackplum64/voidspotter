import torch
from pathlib import Path
import cv2
import os


image_dir = '/home/jackplum/Documents/projects/voidspotter/origdata'

model_path = '/home/jackplum/Documents/projects/yolov5/runs/train/exp28/weights/best.pt'  # Replace with the path to your best.pt model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

model.to('cpu')
device = torch.device('cpu')
print(f"Model is loaded on: {next(model.parameters()).device}")

results = model(original_image, size=640)

# numpy array [x1, y1, x2, y2, confidence, class]
bounding_boxes = results.xyxy[0].cpu().numpy()

for box in bounding_boxes:
    x1, y1, x2, y2, conf, _ = box
    x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
    cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Green color

output_path = Path(output_dir) / image_path.name
cv2.imwrite(str(output_path), img)

print(f'Processed and saved results for {image_path.name}')