import cv2
import argparse
import time

# Thiết lập các tham số dòng lệnh
parser = argparse.ArgumentParser(description="Capture an image using OpenCV and save it with a specified name and format.")
parser.add_argument("-i", "--image", type=str, required=True, help="File name and format for saving the image, e.g., 12345.jpg")
parser.add_argument("-c", "--camera", type=int, default=0, help="Camera index (default is 0 for the primary camera)")
args = parser.parse_args()

# Đường dẫn lưu trữ ảnh
save_path = "/home/pi/License-Plate-Recognition/License_Plate_Picture/" + args.image

# Mở camera
cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    print("Could not open the camera.")
    exit()

# Bắt đầu đếm thời gian
start_time = time.time()
frame = None

# Duy trì mở camera trong 5 giây
while time.time() - start_time < 2:
    ret, frame = cap.read()
    if not ret:
        print("Failed to capture frame.")
        break

# Lưu frame cuối cùng sau 5 giây
if frame is not None:
    # Quay ngược khung hình 180 độ
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_180)
    
    # Lưu ảnh đã quay ngược
    cv2.imwrite(save_path, frame_rotated)
    print(f"Image saved at {save_path}")
else:
    print("No frame captured.")

# Giải phóng camera
cap.release()
cv2.destroyAllWindows()
