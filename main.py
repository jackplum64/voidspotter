import torch
import cv2
import os


def get_bounding_boxes(model, image):
    results = model(image, size=640)
    bounding_boxes = results.xyxy[0].cpu().numpy()  # numpy array [x1, y1, x2, y2, confidence, class]
    return bounding_boxes


def draw_bounding_boxes(model, image):
    boxes = get_bounding_boxes(model, image)

    for box in boxes:
        x1, y1, x2, y2, conf, _ = box
        x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
        cv2.rectangle(image, (x1, y1), (x2, y2), color=(0, 255, 0), thickness=2)  # Green color


def get_model(model_path):
    try:
        model = torch.hub.load('ultralytics/yolov5', 'custom', path=model_path, force_reload=True)
        model.to('cpu')
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
    print(f"Model is loaded on: {next(model.parameters()).device}")


    images_dir = '/home/jackplum/Documents/projects/voidspotter/origdata'
    output_dir = '/home/jackplum/Documents/projects/voidspotter/output'

    images = load_images_from_folder(images_dir)

    for image, filename in images:
        draw_bounding_boxes(model, image)

    save_images_to_folder(images, output_dir)


if __name__ == "__main__":
    main()
