import torch
from pathlib import Path
import cv2
import os

os.environ['CUDA_VISIBLE_DEVICES'] = ''

# Load the YOLOv5 model
model_path = '/home/jackplum/Documents/projects/yolov5/runs/train/exp28/weights/best.pt'  # Replace with the path to your best.pt model
model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)

model.to('cpu')
device = torch.device('cpu')
print(f"Model is loaded on: {next(model.parameters()).device}")

# Directory containing the images
image_dir = '/home/jackplum/Documents/projects/voidspotter/origdata'
output_dir = 'results/'  # Directory to save the results

# Create the output directory if it doesn't exist
os.makedirs(output_dir, exist_ok=True)

# Iterate over all JPG files in the directory
for image_path in Path(image_dir).rglob('*.jpg'):
    # Load an image
    img = cv2.imread(str(image_path))
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

    # Perform inference
    results = model(img_rgb, size=640)

    # Get bounding boxes and scores
    boxes = results.xyxy[0].cpu().numpy()  # [x1, y1, x2, y2, confidence, class]

    # Draw bounding boxes on the image
    for box in boxes:
        x1, y1, x2, y2, conf, _ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(img, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Green color

    # Save the modified image
    output_path = Path(output_dir) / image_path.name
    cv2.imwrite(str(output_path), img)

    print(f'Processed and saved results for {image_path.name}')

print("Inference completed for all images.")
