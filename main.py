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


class ImageClass:
    def __init__(self, image, filename):
        self.image = image
        self.filename = filename
        self.shape = image.shape
        self.bboxes = np.empty((0,6))
        self.classification = None
        self.pass_count = 0

    def classify(self):
        score = (np.sum(self.bboxes[:, 4])) / self.pass_count
        print(f'Filename: {self.filename}')
        print(f'Passcount: {self.pass_count}')
        print(f'Col4 Sum: {np.sum(self.bboxes[:, 4])}')
        print(f'Bbox Count: {self.bboxes.size / 6}')
        print(f'Score: {score}')
        if score <= 0.05:
            self.classification = 'novoid'
        elif score <= 0.15:
            self.classification = 'probably_novoid'
        elif score <= 0.75:
            self.classification = 'maybe_void'
        elif score <= 0.9:
            self.classification = 'probably_void'
        else:
            self.classification = 'void'
        print(f'Class: {self.classification}')
        print()


def get_bounding_boxes(model, image, size):
    results = model(image, size=size)
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


def process_one_pass(model, images):
    for img_obj in images:
        bboxes = get_bounding_boxes(model, img_obj.image, img_obj.shape[0])
        img_obj.bboxes = bboxes
        draw_bounding_boxes(bboxes, img_obj.image, (0, 255, 0))  # color is in (b,g,r)
        img_obj.pass_count += 1


def process_four_cardinal_passes(model, images):
    for img_obj in images:
        rotated_images = rotate_image(img_obj.image)
        colors = [(0, 0, 255), (255, 0, 0), (0, 255, 0), (0, 255, 255)]
        image_size = img_obj.shape[0]

        for itr, (img, rotation_flag) in enumerate(rotated_images):
            bboxes = get_bounding_boxes(model, img, image_size)
            rotated_bboxes = rotate_bounding_boxes(bboxes, img.shape, rotation_flag)
            draw_bounding_boxes(rotated_bboxes, img_obj.image, colors[itr])

            if rotated_bboxes.size != 0:
                img_obj.bboxes = np.concatenate((img_obj.bboxes, rotated_bboxes))
   
        img_obj.pass_count += 4


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
                images.append(ImageClass(img, filename))
    except Exception as e:
        print(f"Error loading images: {e}")
    return images


def save_images_to_folder(images, folder):
    novoid_folder = os.path.join(folder, 'novoid')
    probably_novoid_folder = os.path.join(folder, 'probably_novoid')
    maybe_void_folder = os.path.join(folder, 'maybe_void')
    probably_void_folder = os.path.join(folder, 'probably_void')
    void_folder = os.path.join(folder, 'void')

    if not os.path.exists(novoid_folder):
        os.makedirs(novoid_folder)
    if not os.path.exists(probably_novoid_folder):
        os.makedirs(probably_novoid_folder)
    if not os.path.exists(maybe_void_folder):
        os.makedirs(maybe_void_folder)
    if not os.path.exists(probably_void_folder):
        os.makedirs(probably_void_folder)
    if not os.path.exists(void_folder):
        os.makedirs(void_folder)

    try:
        for img_obj in images:
            if img_obj.classification == 'novoid':
                output_path = os.path.join(novoid_folder, img_obj.filename)
            elif img_obj.classification == 'probably_novoid':
                output_path = os.path.join(probably_novoid_folder, img_obj.filename)
            elif img_obj.classification == 'maybe_void':
                output_path = os.path.join(maybe_void_folder, img_obj.filename)
            elif img_obj.classification == 'probably_void':
                output_path = os.path.join(probably_void_folder, img_obj.filename)
            elif img_obj.classification == 'void':
                output_path = os.path.join(void_folder, img_obj.filename)

            cv2.imwrite(output_path, img_obj.image)
    except Exception as e:
        print(f"Error saving images: {e}")


def main():
    ###os.environ['CUDA_VISIBLE_DEVICES'] = ''
    model_path = '/home/jackplum/Documents/projects/voidspotter/exp20/weights/best.pt'
    model = get_model(model_path)

    images_dir = '/home/jackplum/Documents/projects/voidspotter/outputchops'
    output_dir = '/home/jackplum/Documents/projects/voidspotter/outputchopsclassedALL'

    images = load_images_from_folder(images_dir)

    process_four_cardinal_passes(model, images)

    for img_obj in images:
        img_obj.classify()

    save_images_to_folder(images, output_dir)


if __name__ == "__main__":
    main()
