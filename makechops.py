#!/usr/bin/env python3

import argparse
import pathlib
import os
import numpy as np
import cv2
import pandas as pd
from tqdm import tqdm
import time

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

parser = argparse.ArgumentParser('Image Chopper')
parser.add_argument('inputs', nargs='*')
parser.add_argument('--intermediate', default=None, help='Name of directory to store intermediate outputs. If none, outputs will not be generated')
parser.add_argument('--void-cutoff', default=15, help="Parameter determining void size cutoff")
parser.add_argument('--box-size', default=100, type=int, help="Size (in pixels) of image chops")
parser.add_argument('--x-offset', default=0, type=int, help='Parameter determining how much to offset the mini micrographs in the X direction')
parser.add_argument('--y-offset', default=0, type=int, help='Parameter determining how much to offset the mini micrographs in the Y direction')

args = parser.parse_args()

class record():
    pass
records = []

for input in args.inputs:
    base = str(pathlib.Path(input).absolute()).split('.')[0]
    path_img = pathlib.Path(base+".jpg")
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
    if args.intermediate:
        if not os.path.isdir(args.intermediate): os.mkdir(args.intermediate)
        return "{}/{}_{:05d}_{}.png".format(args.intermediate, rec.stem, rec.tmp_cntr-1, desc)
    return None

## This is the master loop over all input data sets
for rec in records:
    rec.img_micrograph = cv2.imread(str(rec.path_img), 0)
    rec.width = rec.img_micrograph.shape[1]
    rec.height = rec.img_micrograph.shape[0]

    if args.intermediate: cv2.imwrite(getName("Original"), rec.img_micrograph)

    #
    # CHOP AND IDENTIFY THROWAWAY / VOID / NO VOID BOXES
    #
    print("Chopping and classifying micrographs")
    rectangles = []
    x_offset = args.x_offset
    y_offset = args.y_offset

    for x_min in tqdm(range(int(args.box_size / 2) + x_offset, rec.width - args.box_size, args.box_size)):
        for y_min in range(int(args.box_size / 2) + y_offset, rec.height - args.box_size, args.box_size):
            rect = (np.array((x_min, y_min)), np.array((x_min + args.box_size, y_min + args.box_size)))
            rectangles.append(rect)

    if not os.path.isdir("./outputchops"):
        os.mkdir("./outputchops")

    ctr = -1
    for rect in rectangles:
        ctr += 1
        chop_micrograph = rec.img_micrograph[rect[0][1]:rect[1][1], rect[0][0]:rect[1][0]]
        cv2.imwrite("./outputchops/{}_{:05d}_micrograph.png".format(rec.stem, ctr), chop_micrograph)

print(f'Total time taken = {time.perf_counter() - start:0.2f} seconds')
