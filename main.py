import torch
import cv2
import os
import numpy as np
import math
import time

# TO USE:
    # in main(), set model_path to your voidspotter/exp20/weights/best.pt
    # in main(), set images_dir to a folder full of .jpg files you want to run the model on
    # in main(), set output_dir to a folder you wish to save the output files to
    # to force CPU usage, un-comment the two lines with a triple hashtag

def get_bounding_boxes(model, image):
    results = model(image, size=640)
    bounding_boxes = results.xyxy[0].cpu().numpy()  # numpy array [x1, y1, x2, y2, confidence, class]
    return bounding_boxes


def draw_bounding_boxes(bboxes, image, color):
    for box in bboxes:
        x1, y1, x2, y2, conf, _ = box
        color = tuple(conf*val for val in color)
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
        ###model.to('cpu')
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
    print('STARTING')
    t1 = time.perf_counter()
    ###os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_path = '/home/jackplum/Documents/projects/yolov5/runs/train/exp20/weights/best.pt'
    model = get_model(model_path)

    images_dir = '/home/jackplum/Documents/projects/voidspotter/outputchops'
    output_dir = '/home/jackplum/Documents/projects/voidspotter/outputchopsbboxes3'

    images = load_images_from_folder(images_dir)

    t2 = time.perf_counter()
    # Run the ML model once in each direction
    for orig_image, filename in images:
        rotated_images = rotate_image(orig_image)
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]

        for itr, (img, rotation_flag) in enumerate(rotated_images):
            bboxes = get_bounding_boxes(model, img)
            rotated_bboxes = rotate_bounding_boxes(bboxes, img.shape, rotation_flag)
            draw_bounding_boxes(rotated_bboxes, orig_image, colors[itr])

    t3 = time.perf_counter()
    # Run the ML model once in each direction, again
    for orig_image, filename in images:
        rotated_images = rotate_image(orig_image)
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]

        for itr, (img, rotation_flag) in enumerate(rotated_images):
            bboxes = get_bounding_boxes(model, img)
            rotated_bboxes = rotate_bounding_boxes(bboxes, img.shape, rotation_flag)
            draw_bounding_boxes(rotated_bboxes, orig_image, colors[itr])

    # Run the ML model once in the original direction
    #for img, filename in images:
    #    bboxes = get_bounding_boxes(model, img)
    #    draw_bounding_boxes(bboxes, img, (0, 255, 0)) # color is in (b,g,r)
        

    save_images_to_folder(images, output_dir)
    t4 = time.perf_counter()

    print(f'Total Time: {t4 - t1}')
    print(f'Loading Time: {t2 - t1}')
    print(f'First Processing Time: {t3 - t2}')
    print(f'Second Processing Time: {t4 - t3}')


if __name__ == "__main__":
    main()
