# import necessary libraries
import argparse
from genericpath import exists
import os
import torchvision
import torchvision.transforms as T
from pathlib import Path
from runpy import run_path
from PIL import Image
from PIL import Image
from PIL.ExifTags import TAGS
from pathlib import Path
import shutil
from src.piexif import *

parser = argparse.ArgumentParser(
    description='')
parser.add_argument('--device', default='cpu',
                    help="Device to perform inference on 'cpu' or 'gpu'.")
parser.add_argument('--files', help='Input picture directory')
parser.add_argument('--outPut', default='', help='Output picture directory')
parser.add_argument('--addKey', default=True, help='Add XPKeywords for jpeg')

args = parser.parse_args()
# get the pretrained model from torchvision.models
# Note: pretrained=True will get the pretrained weights for the model.
# model.eval() to use the model for inference
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()

# Class labels from official PyTorch documentation for the pretrained model
# Note that there are some N/A's
# for complete list check https://tech.amikelive.com/node-718/what-object-categories-labels-are-in-coco-dataset/
# we will use the same list for this notebook
COCO_INSTANCE_CATEGORY_NAMES = [
    '__background__', 'person', 'bicycle', 'car', 'motorcycle', 'airplane', 'bus',
    'train', 'truck', 'boat', 'traffic light', 'fire hydrant', 'N/A', 'stop sign',
    'parking meter', 'bench', 'bird', 'cat', 'dog', 'horse', 'sheep', 'cow',
    'elephant', 'bear', 'zebra', 'giraffe', 'N/A', 'backpack', 'umbrella', 'N/A', 'N/A',
    'handbag', 'tie', 'suitcase', 'frisbee', 'skis', 'snowboard', 'sports ball',
    'kite', 'baseball bat', 'baseball glove', 'skateboard', 'surfboard', 'tennis racket',
    'bottle', 'N/A', 'wine glass', 'cup', 'fork', 'knife', 'spoon', 'bowl',
    'banana', 'apple', 'sandwich', 'orange', 'broccoli', 'carrot', 'hot dog', 'pizza',
    'donut', 'cake', 'chair', 'couch', 'potted plant', 'bed', 'N/A', 'dining table',
    'N/A', 'N/A', 'toilet', 'N/A', 'tv', 'laptop', 'mouse', 'remote', 'keyboard', 'cell phone',
    'microwave', 'oven', 'toaster', 'sink', 'refrigerator', 'N/A', 'book',
    'clock', 'vase', 'scissors', 'teddy bear', 'hair drier', 'toothbrush'
]


def get_prediction(frame, threshold,gpu=False):
    """
    get_prediction
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
      method:
        - Image is obtained from the image path
        - the image is converted to image tensor using PyTorch's Transforms
        - image is passed through the model to get the predictions
        - class, box coordinates are obtained, but only prediction score > threshold
          are chosen.
    """
    transform = T.Compose([T.ToTensor()])
    img = transform(frame)
    if gpu:
        model.cuda()
        img = img.cuda()
    else:
        model.cpu()
        img = img.cpu()
    pred = model([img])
    pred_class = [COCO_INSTANCE_CATEGORY_NAMES[i]
                  for i in list(pred[0]['labels'].cpu().numpy())]
    pred_boxes = [[(i[0], i[1]), (i[2], i[3])]
                  for i in list(pred[0]['boxes'].cpu().detach().numpy())]
    pred_score = list(pred[0]['scores'].cpu().detach().numpy())
    pred_t = [pred_score.index(x) for x in pred_score if x > threshold]
    if(len(pred_t) == 0):
        return [], []
    pred_t = pred_t[-1]
    pred_boxes = pred_boxes[:pred_t+1]
    pred_class = pred_class[:pred_t+1]
    return pred_boxes, pred_class


def object_detection_api(frame, threshold=0.5, rect_th=3, text_size=3, text_th=3):
    """
    object_detection_api
      parameters:
        - img_path - path of the input image
        - threshold - threshold value for prediction score
        - rect_th - thickness of bounding box
        - text_size - size of the class label text
        - text_th - thichness of the text
      method:
        - prediction is obtained from get_prediction method
        - for each prediction, bounding box is drawn and text is written 
          with opencv
        - the final image is displayed
    """
    boxes, pred_cls = get_prediction(frame, threshold)
    return len(boxes) > 0, pred_cls


def dealImage(path,  outPut='', addKeywords=True, isJpeg=False):
    frame = Image.open(path)
    dectected, pred_cls = object_detection_api(
        frame, threshold=0.8)
    pred_cls_new = []
    [pred_cls_new.append(i) for i in pred_cls if not i in pred_cls_new]
    filename = (Path(path).name)
    if outPut != '':
        filePath = pred_cls_new[0] if(len(pred_cls_new) > 0) else ''
        if not exists(outPut+os.sep+filePath):
            os.makedirs(outPut+os.sep+filePath)
        newPath = outPut+os.sep+filePath+os.sep+filename
        shutil.copy(path, newPath)
        path = newPath
    if isJpeg and addKeywords:
        exif_dict = load(path)
        # for ifd in ("0th", "Exif", "GPS", "1st"):
        #     for tag in exif_dict[ifd]:
        #         print(TAGS[ifd][tag]["name"], exif_dict[ifd][tag])
        pred_cls_new = []
        [pred_cls_new.append(i) for i in pred_cls if not i in pred_cls_new]
        exif_dict['0th'][ImageIFD.XPKeywords] = ';'.join(
            pred_cls_new).encode('utf-16')
        exif_bytes = dump(exif_dict)
        insert(exif_bytes, path)


def last_4chars(x):
    return(x[-4:])


rootdir = Path(args.files)

for subdir, dirs, files in os.walk(rootdir):
    files.sort()
    for file in sorted(files, key=last_4chars):
        print(os.path.join(subdir, file))
        if file.endswith('.mp4'):
            print("It is a video")
            # dealVideo(os.path.join(subdir, file))
        elif file.lower().endswith('.jpg') or file.lower().endswith('.jepg'):
            dealImage(os.path.join(subdir, file), outPut=args.outPut,
                      addKeywords=args.addKey, isJpeg=True)
        elif file.lower().endswith('.png'):
            dealImage(os.path.join(subdir, file), outPut=args.outPut,
                      addKeywords=args.addKey)
