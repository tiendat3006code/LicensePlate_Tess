import cv2
import torch
import argparse
import function.utils_rotate as utils_rotate
import function.helper as helper
from concurrent.futures import ThreadPoolExecutor

# Parse arguments
ap = argparse.ArgumentParser()
ap.add_argument('-i', '--image', required=True, help='path to input image')
args = ap.parse_args()

# Load YOLO models only once
if not hasattr(helper, 'yolo_LP_detect_loaded'):
    yolo_LP_detect = torch.hub.load('yolov5', 'custom', path='model/LP_detector.pt', force_reload=True, source='local')
    helper.yolo_LP_detect_loaded = True
if not hasattr(helper, 'yolo_license_plate_loaded'):
    yolo_license_plate = torch.hub.load('yolov5', 'custom', path='model/LP_ocr.pt', force_reload=True, source='local')
    helper.yolo_license_plate_loaded = True
yolo_license_plate.conf = 0.60

# Read the image
img = cv2.imread(args.image)

# Detect license plates
plates = yolo_LP_detect(img, size=640)
list_plates = plates.pandas().xyxy[0].values.tolist()
list_read_plates = set()

if len(list_plates) == 0:
    # If no plates are detected, try to read plate directly
    lp = helper.read_plate(yolo_license_plate, img)
    if lp != "unknown":
        list_read_plates.add(lp)
else:
    # Process detected plates
    with ThreadPoolExecutor() as executor:
        futures = []
        for plate in list_plates:
            x = int(plate[0])  # xmin
            y = int(plate[1])  # ymin
            w = int(plate[2] - plate[0])  # xmax - xmin
            h = int(plate[3] - plate[1])  # ymax - ymin  
            crop_img = img[y:y+h, x:x+w]
            for cc in range(2):
                for ct in range(2):
                    futures.append(executor.submit(helper.read_plate, yolo_license_plate, utils_rotate.deskew(crop_img, cc, ct)))
        
        for future in futures:
            lp = future.result()
            if lp != "unknown":
                list_read_plates.add(lp)

# Print out detected license plates
if list_read_plates:
    for lp in list_read_plates:
        print(lp)
else:
    print("0000")
