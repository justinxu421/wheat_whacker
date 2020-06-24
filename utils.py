"""Utility functions for loading, processing, and augmenting data"""

import pandas as pd
import numpy as np
import os
import re
import random
import math
import cv2
from PIL import Image

import imageio
import imgaug as ia
from imgaug import augmenters as iaa 
from imgaug.augmentables.bbs import BoundingBox, BoundingBoxesOnImage

import torch
import torchvision

from torchvision.models.detection.faster_rcnn import FastRCNNPredictor
from torchvision.models.detection import FasterRCNN
from torchvision.models.detection.rpn import AnchorGenerator

from torch.utils.data import Dataset
from torch.utils.data.sampler import SequentialSampler

def expand_bbox(x):
    r = np.array(re.findall("([0-9]+[.]?[0-9]*)", x))
    if len(r) == 0:
        r = [-1, -1, -1, -1]
    return r


def load_data(DIR_INPUT, load = False):
    """
    Load and split data into train, val, test sets
    Params: 
        load (bool) - load in train/val splits from file
    """
    # Load raw data into dataframes
    train_df = pd.read_csv(f'{DIR_INPUT}/train.csv')
    test_df = pd.read_csv(f'{DIR_INPUT}/sample_submission.csv')
    train_df.shape

    # Initialize each bounding box sample in XYWH format
    train_df['x'] = -1
    train_df['y'] = -1
    train_df['w'] = -1
    train_df['h'] = -1

    # Fill each sample from column 'bbox'
    train_df[['x', 'y', 'w', 'h']] = np.stack(train_df['bbox'].apply(lambda x: expand_bbox(x)))
    train_df.drop(columns=['bbox'], inplace=True)
    train_df['x'] = train_df['x'].astype(np.float)
    train_df['y'] = train_df['y'].astype(np.float)
    train_df['w'] = train_df['w'].astype(np.float)
    train_df['h'] = train_df['h'].astype(np.float)
    
    # Split train-val from saved splits if load
    if load:
        train_ids = np.load('train_ids.npy', allow_pickle = True)
        valid_ids = np.load('valid_ids.npy', allow_pickle = True)
    else:
        image_ids = train_df['image_id'].unique()
        valid_ids = image_ids[-665:]
        train_ids = image_ids[:-665]

    valid_df = train_df[train_df['image_id'].isin(valid_ids)]
    train_df = train_df[train_df['image_id'].isin(train_ids)]

    return train_df, valid_df, test_df

class WheatDataset(Dataset):
    """Class for loading and transforming training images
    """
    def __init__(self, dataframe, image_dir, augment=True):
        super().__init__()
        
        # Store dataset dataframe and each column
        self.df = dataframe
        self.image_ids = dataframe['image_id'].unique()

       # Store image directory
        self.image_dir = image_dir
        self.img_size = 1024
        
        # Store augmentation flags
        self.augment = augment

    def __getitem__(self, index: int):
        # Get image and box coordinates
        if self.augment and random.random() > 0.5:
            image, coords = self.load_cutmix_image_and_boxes(index)
        else:
            image, coords = self.load_image(index)

        if self.augment:
            # Get bounding boxes
            boxes = [BoundingBox(coord[0], coord[1], coord[2], coord[3]) for coord in coords]
            bbs = BoundingBoxesOnImage(boxes, (self.img_size, self.img_size))

            # Apply augmentation
            blur_k = 5
            
            seq = iaa.Sequential([
#                 iaa.CoarseDropout((0.0, 0.05), size_percent=(0.02, 0.25)),
                # wind caused blur
                iaa.MotionBlur(k=blur_k, angle=[-45, 45]), 
                # camera perspectives
#                 iaa.ShearX((-20, 20)), 
#                 iaa.ShearY((-20, 20)),
                iaa.Rot90((0, 3), keep_size=False),
                iaa.Fliplr(0.5),
                iaa.Flipud(0.5),
                # color variety of wheat # iaa.AddToHueAndSaturation((-60, 60)),
                iaa.ChangeColorTemperature((4000, 20000)),
                iaa.AddToBrightness((-30, 50)),
                iaa.Cutout(fill_mode='constant', size=0.1, cval=0),
            ])
            
            # TODO: hyper parameter for augmentations?
            # TODO: water-like effect
            # iaa.ElasticTransformation(alpha=90, sigma=9),   
            if random.random() > 0.5:
                image, bbs = seq(image=image, bounding_boxes=bbs)
                
            coords = bbs.to_xyxy_array()
        
        # Return augmented image and bounding boxes as tensors
        image = image.astype(np.float32) / 255.0 # .permute(2, 0, 1)
        image = torch.from_numpy(image).permute(2,0,1)
        d = {
            'boxes': torch.from_numpy(coords.astype(np.float32)),
            'labels': torch.ones((coords.shape[0],), dtype=torch.int64)
        }
        return image, d

    def __len__(self) -> int:
        return self.image_ids.shape[0]
    
    def load_image(self, index):
        """
        Loads an image from self.image_ids
        """
        image_id = self.image_ids[index]
        imgpath = f'{self.image_dir}'
        img = cv2.imread(f'{imgpath}/{image_id}.jpg', cv2.IMREAD_COLOR)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.uint8)
        
        records = self.df[self.df['image_id'] == image_id]
        boxes = records[['x', 'y', 'w', 'h']].values
        boxes[:, 2] = boxes[:, 0] + boxes[:, 2]
        boxes[:, 3] = boxes[:, 1] + boxes[:, 3]

        assert img is not None, 'Image Not Found ' + imgpath
        return img, boxes

    def load_cutmix_image_and_boxes(self, index, imsize=1024):
        """ 
        This implementation of cutmix author:  https://www.kaggle.com/nvnnghia 
        Refactoring and adaptation: https://www.kaggle.com/shonenkov
        """
        w, h = imsize, imsize
        s = imsize // 2
    
        xc, yc = [int(random.uniform(imsize * 0.25, imsize * 0.75)) for _ in range(2)]  # center x, y
        indexes = [index] + [random.randint(0, self.image_ids.shape[0] - 1) for _ in range(3)]

        result_image = np.full((imsize, imsize, 3), 1, dtype=np.uint8)
        result_boxes = []

        for i, index in enumerate(indexes):
            image, boxes = self.load_image(index)
            if i == 0:
                x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
            elif i == 1:  # top right
                x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
                x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
            elif i == 2:  # bottom left
                x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
            elif i == 3:  # bottom right
                x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
                x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)
            result_image[y1a:y2a, x1a:x2a] = image[y1b:y2b, x1b:x2b]
            padw = x1a - x1b
            padh = y1a - y1b

            boxes[:, 0] += padw
            boxes[:, 1] += padh
            boxes[:, 2] += padw
            boxes[:, 3] += padh

            result_boxes.append(boxes)

        result_boxes = np.concatenate(result_boxes, 0)
        np.clip(result_boxes[:, 0:], 0, 2 * s, out=result_boxes[:, 0:])
        result_boxes = result_boxes.astype(np.int32)
        result_boxes = result_boxes[np.where((result_boxes[:,2]-result_boxes[:,0])*(result_boxes[:,3]-result_boxes[:,1]) > 0)]
        return result_image, result_boxes
    
class Averager:
    """Records and averages the model loss per iteration
    """
    def __init__(self):
        self.current_total = 0.0
        self.iterations = 0.0

    def send(self, value):
        self.current_total += value
        self.iterations += 1

    @property
    def value(self):
        if self.iterations == 0:
            return 0
        else:
            return 1.0 * self.current_total / self.iterations

    def reset(self):
        self.current_total = 0.0
        self.iterations = 0.0
    
    
def format_prediction_string(boxes, scores):
    """Format bounding box predictions into a string for submission
    """
    pred_strings = []
    for j in zip(scores, boxes):
        pred_strings.append("{0:.4f} {1} {2} {3} {4}".format(j[0], j[1][0], j[1][1], j[1][2], j[1][3]))

    return " ".join(pred_strings)
    
    
def predict_bbox(images, image_ids, model, detection_threshold, submission=False):
    """Predicts bounding boxes for a list of images if they exceed the score 
    threshold, and returns as a dataframe formatted for further training or submission.
    """
    
    outputs = model(images)
    results = []
    
    for i, image in enumerate(images):
        # Predict boxes
        boxes = outputs[i]['boxes'].data.cpu().numpy()
        scores = outputs[i]['scores'].data.cpu().numpy()
        
        # Cut off at probability score threshold
        boxes = boxes[scores >= detection_threshold].astype(np.int32)
        scores = scores[scores >= detection_threshold]
        image_id = image_ids[i]
        
        # Format results
        if submission == True: 
            boxes[:, 2] = boxes[:, 2] - boxes[:, 0]
            boxes[:, 3] = boxes[:, 3] - boxes[:, 1]
            results.append({
                'image_id': image_id,
                'PredictionString': format_prediction_string(boxes, scores)
            })
        else:
            for box in boxes:
                results.append({
                    'image_id': image_id,
                    'x1': box[0],
                    'y1': box[1],
                    'x2': box[2],
                    'y2': box[3],
                })
    return results

    

### Old augmentations replaced with imgaug library
    
# def load_mosaic(self, index):
#     """
#     Applies mosaic mix and match, then a random affine, to an image and its labels.
#     Specifically, creates a 2x2 image tile, affine transforms it, then takes a center
#     crop using negative borders.
#     """

#     labels4 = []
#     s = self.img_size
#     xc, yc = [int(random.uniform(s * 0.5, s * 1.5)) for _ in range(2)]  # mosaic center x, y
#     indices = [index] + [random.randint(0, len(self.labels) - 1) for _ in range(3)]  # 3 additional image indices
#     for i, index in enumerate(indices):
#         # Load image
#         img, _, (h, w) = load_image(self, index)

#         # place img in img4
#         if i == 0:  # top left
#             img4 = np.full((s * 2, s * 2, img.shape[2]), 114, dtype=np.uint8)  # base image with 4 tiles
#             x1a, y1a, x2a, y2a = max(xc - w, 0), max(yc - h, 0), xc, yc  # xmin, ymin, xmax, ymax (large image)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), h - (y2a - y1a), w, h  # xmin, ymin, xmax, ymax (small image)
#         elif i == 1:  # top right
#             x1a, y1a, x2a, y2a = xc, max(yc - h, 0), min(xc + w, s * 2), yc
#             x1b, y1b, x2b, y2b = 0, h - (y2a - y1a), min(w, x2a - x1a), h
#         elif i == 2:  # bottom left
#             x1a, y1a, x2a, y2a = max(xc - w, 0), yc, xc, min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = w - (x2a - x1a), 0, max(xc, w), min(y2a - y1a, h)
#         elif i == 3:  # bottom right
#             x1a, y1a, x2a, y2a = xc, yc, min(xc + w, s * 2), min(s * 2, yc + h)
#             x1b, y1b, x2b, y2b = 0, 0, min(w, x2a - x1a), min(y2a - y1a, h)

#         img4[y1a:y2a, x1a:x2a] = img[y1b:y2b, x1b:x2b]  # img4[ymin:ymax, xmin:xmax]
#         padw = x1a - x1b
#         padh = y1a - y1b

#         # Labels
#         x = self.labels[index]
#         labels = x.copy()
#         if x.size > 0:  # Normalized xywh to pixel xyxy format
#             labels[:, 1] = w * (x[:, 1] - x[:, 3] / 2) + padw
#             labels[:, 2] = h * (x[:, 2] - x[:, 4] / 2) + padh
#             labels[:, 3] = w * (x[:, 1] + x[:, 3] / 2) + padw
#             labels[:, 4] = h * (x[:, 2] + x[:, 4] / 2) + padh
#         labels4.append(labels)

#     # Concat/clip labels
#     if len(labels4):
#         labels4 = np.concatenate(labels4, 0)
#         # np.clip(labels4[:, 1:] - s / 2, 0, s, out=labels4[:, 1:])  # use with center crop
#         np.clip(labels4[:, 1:], 0, 2 * s, out=labels4[:, 1:])  # use with random_affine

#     # Augment
#     # img4 = img4[s // 2: int(s * 1.5), s // 2:int(s * 1.5)]  # center crop (WARNING, requires box pruning)
#     img4, labels4 = random_affine(img4, labels4,
#                                   degrees=1.98 * 2,
#                                   translate=0.05 * 2,
#                                   scale=0.05 * 2,
#                                   shear=0.641 * 2,
#                                   border=-s // 2)  # border to remove

#     return img4, labels4

# def random_affine(img, targets=(), degrees=10, translate=.1, scale=.1, shear=10, border=0):
#     """Applies a random affine transformation that preserves parallel lines, but not angles,
#     for both the img and bounding box targets

#     torchvision.transforms.RandomAffine(degrees=(-10, 10), translate=(.1, .1), scale=(.9, 1.1), shear=(-10, 10))
#     https://medium.com/uruvideo/dataset-augmentation-with-random-homographies-a8f4b44830d4
#     """

#     # List of bounding boxes to similarly transform
#     if targets is None:  # targets = [cls, xyxy]
#         targets = []

#     # New image height width after border adjustment
#     height = img.shape[0] + border * 2
#     width = img.shape[1] + border * 2

#     # Rotation and Scale
#     R = np.eye(3)
#     a = random.uniform(-degrees, degrees)
#     # a += random.choice([-180, -90, 0, 90])  # add 90deg rotations to small rotations
#     s = random.uniform(1 - scale, 1 + scale)
#     R[:2] = cv2.getRotationMatrix2D(angle=a, center=(img.shape[1] / 2, img.shape[0] / 2), scale=s)

#     # Translation
#     T = np.eye(3)
#     T[0, 2] = random.uniform(-translate, translate) * img.shape[0] + border  # x translation (pixels)
#     T[1, 2] = random.uniform(-translate, translate) * img.shape[1] + border  # y translation (pixels)

#     # Shear
#     S = np.eye(3)
#     S[0, 1] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # x shear (deg)
#     S[1, 0] = math.tan(random.uniform(-shear, shear) * math.pi / 180)  # y shear (deg)

#     # Combined rotation matrix
#     M = S @ T @ R  # ORDER IS IMPORTANT HERE!!
#     if (border != 0) or (M != np.eye(3)).any():  # image changed
#         img = cv2.warpAffine(img, M[:2], dsize=(width, height), flags=cv2.INTER_LINEAR, borderValue=(114, 114, 114))

#     # Transform label coordinates
#     n = len(targets)
#     if n:
#         # warp points
#         xy = np.ones((n * 4, 3))
#         xy[:, :2] = targets[:, [1, 2, 3, 4, 1, 4, 3, 2]].reshape(n * 4, 2)  # x1y1, x2y2, x1y2, x2y1
#         xy = (xy @ M.T)[:, :2].reshape(n, 8)

#         # create new boxes
#         x = xy[:, [0, 2, 4, 6]]
#         y = xy[:, [1, 3, 5, 7]]
#         xy = np.concatenate((x.min(1), y.min(1), x.max(1), y.max(1))).reshape(4, n).T

#         # # apply angle-based reduction of bounding boxes
#         # radians = a * math.pi / 180
#         # reduction = max(abs(math.sin(radians)), abs(math.cos(radians))) ** 0.5
#         # x = (xy[:, 2] + xy[:, 0]) / 2
#         # y = (xy[:, 3] + xy[:, 1]) / 2
#         # w = (xy[:, 2] - xy[:, 0]) * reduction
#         # h = (xy[:, 3] - xy[:, 1]) * reduction
#         # xy = np.concatenate((x - w / 2, y - h / 2, x + w / 2, y + h / 2)).reshape(4, n).T

#         # reject warped points outside of image
#         xy[:, [0, 2]] = xy[:, [0, 2]].clip(0, width)
#         xy[:, [1, 3]] = xy[:, [1, 3]].clip(0, height)
#         w = xy[:, 2] - xy[:, 0]
#         h = xy[:, 3] - xy[:, 1]
#         area = w * h
#         area0 = (targets[:, 3] - targets[:, 1]) * (targets[:, 4] - targets[:, 2])
#         ar = np.maximum(w / (h + 1e-16), h / (w + 1e-16))  # aspect ratio
#         i = (w > 4) & (h > 4) & (area / (area0 * s + 1e-16) > 0.2) & (ar < 10)

#         targets = targets[i]
#         targets[:, 1:5] = xy[i]

#     return img, targets

# def augment_hsv(img, hgain=0.5, sgain=0.5, vgain=0.5):
#     """Applies random hue saturation value gains to the image.
#     """

#     r = np.random.uniform(-1, 1, 3) * [hgain, sgain, vgain] + 1  # random gains
#     hue, sat, val = cv2.split(cv2.cvtColor(img, cv2.COLOR_BGR2HSV))
#     dtype = img.dtype  # uint8

#     x = np.arange(0, 256, dtype=np.int16)
#     lut_hue = ((x * r[0]) % 180).astype(dtype)
#     lut_sat = np.clip(x * r[1], 0, 255).astype(dtype)
#     lut_val = np.clip(x * r[2], 0, 255).astype(dtype)

#     img_hsv = cv2.merge((cv2.LUT(hue, lut_hue), cv2.LUT(sat, lut_sat), cv2.LUT(val, lut_val))).astype(dtype)
#     cv2.cvtColor(img_hsv, cv2.COLOR_HSV2BGR, dst=img)  # no return needed

# def letterbox(img, new_shape=(416, 416), color=(114, 114, 114), auto=True, scaleFill=False, scaleup=True):
#     """Downsizes and pads image dimensions to be 32-pixel multiples for faster training
#     on mixed aspect ratio data (according to https://github.com/ultralytics/yolov3/issues/232) 
#     """

#     shape = img.shape[:2]  # current shape [height, width]
#     if isinstance(new_shape, int):
#         new_shape = (new_shape, new_shape)

#     # Scale ratio (new / old)
#     r = min(new_shape[0] / shape[0], new_shape[1] / shape[1])
#     if not scaleup:  # only scale down, do not scale up (for better test mAP)
#         r = min(r, 1.0)

#     # Compute padding
#     ratio = r, r  # width, height ratios
#     new_unpad = int(round(shape[1] * r)), int(round(shape[0] * r))
#     dw, dh = new_shape[1] - new_unpad[0], new_shape[0] - new_unpad[1]  # wh padding
#     if auto:  # minimum rectangle
#         dw, dh = np.mod(dw, 64), np.mod(dh, 64)  # wh padding
#     elif scaleFill:  # stretch
#         dw, dh = 0.0, 0.0
#         new_unpad = new_shape
#         ratio = new_shape[0] / shape[1], new_shape[1] / shape[0]  # width, height ratios

#     dw /= 2  # divide padding into 2 sides
#     dh /= 2

#     if shape[::-1] != new_unpad:  # resize
#         img = cv2.resize(img, new_unpad, interpolation=cv2.INTER_LINEAR)
#     top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
#     left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
#     img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # add border
#     return img, ratio, (dw, dh)

    
