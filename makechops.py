#!/usr/bin/env python3

import pathlib
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import time

# Configuration
inputs = [
    "origdata/1_001", "origdata/1_002", "origdata/1_003", "origdata/1_004", "origdata/1_005", "origdata/1_006", "origdata/1_007", "origdata/1_008", "origdata/1_009", "origdata/1_010",
    "origdata/2_001", "origdata/2_002", "origdata/2_003", "origdata/2_004", "origdata/2_005", "origdata/2_006", "origdata/2_007", "origdata/2_008", "origdata/2_009",
    "origdata/3_001", "origdata/3_002", "origdata/3_003", "origdata/3_004", "origdata/3_005", "origdata/3_006", "origdata/3_007", "origdata/3_008", "origdata/3_009", "origdata/3_010",
    "origdata/4_001", "origdata/4_002", "origdata/4_003", "origdata/4_004", "origdata/4_005", "origdata/4_006", "origdata/4_007", "origdata/4_008", "origdata/4_009",
    "origdata/5_001", "origdata/5_002", "origdata/5_003", "origdata/5_004", "origdata/5_005", "origdata/5_006", "origdata/5_007", "origdata/5_008", "origdata/5_009", "origdata/5_010", "origdata/5_011",
    "origdata/6_001", "origdata/6_002", "origdata/6_003", "origdata/6_004", "origdata/6_005", "origdata/6_006", "origdata/6_007", "origdata/6_008", "origdata/6_009", "origdata/6_010"
]
intermediate = None  # Name of directory to store intermediate outputs. If none, outputs will not be generated
void_cutoff = 15  # Parameter determining void size cutoff
box_size = 100  # Size (in pixels) of image chops
x_offset = 0  # Parameter determining how much to offset the mini micrographs in the X direction
y_offset = 0  # Parameter determining how much to offset the mini micrographs in the Y direction

def doesRectangleOverlap(rect, x1, y1, x2, y2):
    box_x1, box_y1 = rect[0]
    box_x2, box_y2 = rect[1]

    if x1 >= box_x2 or x2 <= box_x1:
        return False

    if y1 >= box_y2 or y2 <= box_y1:
        return False

    return True

def integral_phi(rect, bounding_boxes):
    x_min, y_min = rect[0]
    x_max, y_max = rect[1]
    dx, dy = 1, 1  # step size for integration
    integral = 0
    for x in np.arange(x_min, x_max, dx):
        for y in np.arange(y_min, y_max, dy):
            integral += phi(x, y, bounding_boxes) * dx * dy
    return integral

def phi(x, y, bounding_boxes):
    combined_value = 0
    for box in bounding_boxes:
        x1, y1, x2, y2 = box[:4]
        if x1 <= x <= x2 and y1 <= y <= y2:
            combined_value += 1  # Assuming a constant value for simplicity; adjust as needed
    return combined_value

start = time.perf_counter()

class record():
    pass
records = []

for input in inputs:
    base = str(pathlib.Path(input).absolute()).split('.')[0]
    path_img = pathlib.Path(base + ".jpg")
    if path_img.exists():
        rec = record()
        rec.stem = path_img.stem
        rec.base = base
        rec.path_img = path_img
        rec.tmp_cntr = 0
        if rec.stem not in [r.stem for r in records]:
            records.append(rec)

def getName(desc):
    rec.tmp_cntr += 1
    if intermediate:
        if not os.path.isdir(intermediate): os.mkdir(intermediate)
        return "{}/{}_{:05d}_{}.png".format(intermediate, rec.stem, rec.tmp_cntr - 1, desc)
    return None

## This is the master loop over all input data sets
for rec in records:
    rec.img_micrograph = cv2.imread(str(rec.path_img), 0)
    rec.width = rec.img_micrograph.shape[1]
    rec.height = rec.img_micrograph.shape[0]

    if intermediate: cv2.imwrite(getName("Original"), rec.img_micrograph)

    #
    # CHOP AND IDENTIFY THROWAWAY / VOID / NO VOID BOXES
    #
    print("Chopping and classifying micrographs")
    rectangles = []

    for x_min in tqdm(range(int(box_size / 2) + x_offset, rec.width - box_size, box_size)):
        for y_min in range(int(box_size / 2) + y_offset, rec.height - box_size, box_size):
            rect = (np.array((x_min, y_min)), np.array((x_min + box_size, y_min + box_size)))
            rectangles.append(rect)

    if not os.path.isdir("./outputchops"):
        os.mkdir("./outputchops")

    ctr = -1
    for rect in rectangles:
        ctr += 1
        chop_micrograph = rec.img_micrograph[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        cv2.imwrite("./outputchops/{}_{:05d}_micrograph.png".format(rec.stem, ctr), chop_micrograph)

print(f'Total time taken = {time.perf_counter() - start:0.2f} seconds')
