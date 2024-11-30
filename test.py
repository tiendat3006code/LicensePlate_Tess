from PIL import Image
import pytesseract

# Đường dẫn Tesseract (Chỉ cần trên Windows)
# pytesseract.pytesseract.tesseract_cmd = r"C:\Program Files\Tesseract-OCR\tesseract.exe"

# Mở ảnh
image = Image.open('/home/pi/License-Plate-Recognition/License_Plate_Picture/23_11_2024_11_08_25.jpg')

# Thực hiện OCR
text = pytesseract.image_to_string(image, config='--psm 6').strip()

print("Extracted Text:")
print(text)
