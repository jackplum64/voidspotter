import torch
import cv2
import os
import numpy as np

def draw_multiple_bounding_boxes(model, orig_image):
    images = rotate_image(orig_image)
    colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]

    for itr, (img, rotation_flag) in enumerate(images):
        bboxes = get_bounding_boxes(model, img)
        rotated_bboxes = rotate_bounding_boxes(bboxes, img.shape, rotation_flag)
        draw_bounding_boxes(rotated_bboxes, orig_image, colors[itr])



def get_bounding_boxes(model, image):
    results = model(image, size=640)
    bounding_boxes = results.xyxy[0].cpu().numpy()  # numpy array [x1, y1, x2, y2, confidence, class]
    return bounding_boxes


def draw_bounding_boxes(bboxes, image, color):
    for box in bboxes:
        x1, y1, x2, y2, conf, _ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=color, thickness=1)
    

def rotate_image(image):
    image_90 = cv2.rotate(image, cv2.ROTATE_90_CLOCKWISE)
    image_180 = cv2.rotate(image, cv2.ROTATE_180)
    image_270 = cv2.rotate(image, cv2.ROTATE_90_COUNTERCLOCKWISE)
    
    return [
        (image, None),
        (image_90, cv2.ROTATE_90_CLOCKWISE),
        (image_180, cv2.ROTATE_180),
        (image_270, cv2.ROTATE_90_COUNTERCLOCKWISE)
    ]

    
def rotate_bounding_boxes(bboxes, image_shape, rotation_flag):
    h, w = image_shape[:2]
    rotated_bboxes = []

    for bbox in bboxes:
        x1, y1, x2, y2, confidence, cls = bbox
        
        if rotation_flag == cv2.ROTATE_90_CLOCKWISE:
            new_x1 = y1
            new_y1 = w - x2
            new_x2 = y2
            new_y2 = w - x1
        
        elif rotation_flag == cv2.ROTATE_180:
            new_x1 = w - x2
            new_y1 = h - y2
            new_x2 = w - x1
            new_y2 = h - y1
        
        elif rotation_flag == cv2.ROTATE_90_COUNTERCLOCKWISE:
            new_x1 = h - y2
            new_y1 = x1
            new_x2 = h - y1
            new_y2 = x2

        elif rotation_flag is None:
            new_x1 = x1
            new_y1 = y1
            new_x2 = x2
            new_y2 = y2
        
        else:
            raise ValueError("Invalid rotation_flag. Use cv2.ROTATE_90_CLOCKWISE, cv2.ROTATE_180, or cv2.ROTATE_90_COUNTERCLOCKWISE.")
        
        rotated_bboxes.append([new_x1, new_y1, new_x2, new_y2, confidence, cls])
    
    return np.array(rotated_bboxes)




def get_model(model_path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.to('cpu')
        print(f"Model is loaded on: {next(model.parameters()).device}")
        return model
    except Exception as e:
        print(f"Error loading model: {e}")
        return None


def load_images_from_folder(folder):
    images = []
    try:
        for filename in os.listdir(folder):
            img_path = os.path.join(folder, filename)
            img = cv2.imread(img_path)
            if img is not None:
                images.append((img, filename))
    except Exception as e:
        print(f"Error loading images: {e}")
    return images


def save_images_to_folder(images, folder):
    if not os.path.exists(folder):
        os.makedirs(folder)
    try:
        for i, (image, filename) in enumerate(images):
            output_path = os.path.join(folder, filename)
            cv2.imwrite(output_path, image)
    except Exception as e:
        print(f"Error saving images: {e}")


def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_path = '/home/jackplum/Documents/projects/yolov5/runs/train/exp20/weights/best.pt'
    model = get_model(model_path)

    images_dir = '/home/jackplum/Documents/projects/voidspotter/origdata'
    output_dir = '/home/jackplum/Documents/projects/voidspotter/output'

    images = load_images_from_folder(images_dir)

    for image, filename in images:
        draw_multiple_bounding_boxes(model, image)

    save_images_to_folder(images, output_dir)


if __name__ == "__main__":
    main()
