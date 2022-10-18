import argparse

parser = argparse.ArgumentParser(description='hide faces')
parser.add_argument('-m', default='retinanet', help='model to use (retinanet/yolov3)')
parser.add_argument('in_file', help='path to input image', metavar='input_path')
args = parser.parse_args()

if(args.m == 'retinanet'):
    config_file = 'models/retinanet/retinanet_r50_fpn_fp16_1x_coco.py'
    checkpoint_file = 'models/retinanet/retinanet_r50_fpn_fp16_1x_coco_20200702-0dbfb212.pth'
elif(args.m == 'yolov3'):
    config_file = 'models/yolo3/yolov3_mobilenetv2_320_300e_coco.py'
    checkpoint_file = 'models/yolo3/yolov3_mobilenetv2_320_300e_coco_20210719_215349-d18dff72.pth'
else:
    raise Exception("invalid model input -- retinanet/yolov3 only")

from mmdet.apis import init_detector, inference_detector
from mmcv.runner import load_checkpoint
import cv2
import numpy as np
import torch

import time

global resize
resize = 100

dev = 'cuda:0' if torch.cuda.is_available() else 'cpu'

obdet_model = init_detector(config_file, checkpoint_file, device=dev)

face_model = cv2.FaceDetectorYN.create(
    "models/face_detection_yunet_2022mar.onnx",
    "",
    (resize, resize),
    0.3,
    0.4,
    20
)

def get_img_from_bb(img, bounding_box):
    return img[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])]

def detect_face_on_person(person_img):
    orig_shape = person_img.shape
    person_img = cv2.resize(person_img, (resize, resize))

    faces = face_model.detect(person_img)

    if(faces[1] is not None):
        for face in faces[1]:
            coords = face[0:4].astype(np.int32)
            cv2.rectangle(person_img, (coords[0], coords[1]), (coords[0]+coords[2], coords[1]+coords[3]), (0, 0, 0), -1)
    # print(orig_shape)
    person_img = cv2.resize(person_img, (orig_shape[1], orig_shape[0]))
    return person_img

def replace_person(img, bounding_box, new_img):
    img[int(bounding_box[1]):int(bounding_box[3]), int(bounding_box[0]):int(bounding_box[2])] = new_img
    return img

def hide_faces(img):
    result = inference_detector(obdet_model, img)

    classes = obdet_model.CLASSES
    person_bbs = result[classes.index('person')]
    # print(person_bb)

    for person in person_bbs:
        if(person[-1] < 0.2):
            continue
        single_img = get_img_from_bb(img, person)
        single_img = detect_face_on_person(single_img)
        img = replace_person(img, person, single_img)

    return img

start = time.time()
img = cv2.imread(args.in_file)

img = hide_faces(img)

cv2.imwrite('demo/result.jpg', img)
