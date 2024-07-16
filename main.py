import torch
from torchvision import transforms
from PIL import Image, ImageDraw
import pathlib
import os
import cv2

# Load the model
model = torch.jit.load("/home/jackplum/Documents/projects/yolov5/runs/train/exp3/weights/best.torchscript")

img = cv2.imread("/home/jackplum/Documents/projects/voidspotter/origdata/1_001.jpg")[..., ::-1]

img = img.copy()

transform = transforms.ToTensor()
img_tensor = transform(img)

img_tensor = img_tensor.unsqueeze(0)

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
img_tensor = img_tensor.to(device)

results = model(img)

print(results.pandas().xyxy[0])