import json
import os

def polygon_to_bbox(polygon):
    x_coords = [point[0] for point in polygon]
    y_coords = [point[1] for point in polygon]
    min_x = min(x_coords)
    max_x = max(x_coords)
    min_y = min(y_coords)
    max_y = max(y_coords)
    return min_x, min_y, max_x, max_y

def convert_annotations(input_dir, output_dir):
    for filename in os.listdir(input_dir):
        if filename.endswith(".json"):
            with open(os.path.join(input_dir, filename)) as f:
                data = json.load(f)
            annotations = data['annotations']
            with open(os.path.join(output_dir, filename.replace('.json', '.txt')), 'w') as out_file:
                for annotation in annotations:
                    polygon = annotation['segmentation'][0]  # assuming single polygon per annotation
                    bbox = polygon_to_bbox(polygon)
                    class_id = annotation['category_id']
                    center_x = (bbox[0] + bbox[2]) / 2
                    center_y = (bbox[1] + bbox[3]) / 2
                    width = bbox[2] - bbox[0]
                    height = bbox[3] - bbox[1]
                    out_file.write(f"{class_id} {center_x} {center_y} {width} {height}\n")

input_directory = "/home/jackplum/Documents/projects/voidspotter/data/labels/polygon/"
output_directory = "//home/jackplum/Documents/projects/voidspotter/data/labels/train/"
convert_annotations(input_directory, output_directory)