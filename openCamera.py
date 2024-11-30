import cv2
import argparse

# Thiết lập đối số dòng lệnh
parser = argparse.ArgumentParser(description="Chương trình hiển thị webcam với camera tùy chỉnh.")
parser.add_argument('-c', '--camera', type=int, default=0, help='Chỉ số camera (mặc định: 0)')
args = parser.parse_args()

# Mở camera dựa trên tham số đầu vào
cap = cv2.VideoCapture(args.camera)

if not cap.isOpened():
    print(f"Không thể mở camera với chỉ số {args.camera}")
    exit()

while True:
    # Đọc một frame từ camera
    ret, frame = cap.read()

    # Kiểm tra xem frame có được đọc thành công không
    if not ret:
        print("Không thể nhận frame, kết thúc")
        break

    # Quay ngược frame 180 độ
    frame_rotated = cv2.rotate(frame, cv2.ROTATE_180)

    # Hiển thị frame đã quay
    cv2.imshow('Webcam', frame_rotated)

    # Nhấn phím 'q' để thoát
    if cv2.waitKey(1) & 0xFF == ord('q'):
        # Lưu ảnh cuối cùng nếu cần
        cv2.imwrite("captured_image.jpg", frame_rotated)
        break

# Giải phóng camera và đóng các cửa sổ
cap.release()
cv2.destroyAllWindows()
