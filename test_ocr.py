import pytesseract
from PIL import Image
import cv2
import numpy as np
import argparse
import re

pytesseract.pytesseract.tesseract_cmd = "/usr/bin/tesseract"

def parse_args():
    parser = argparse.ArgumentParser(description="OCR License Plate Recognition")
    parser.add_argument('-i', '--image', required=True, help="Path to the image file")
    parser.add_argument('-o', '--output', default="output.jpg", help="Path to save the output image with boxes")
    return parser.parse_args()

def detect_text_regions(image_path):
    """Phát hiện các vùng nghi ngờ có chữ."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold để xử lý ánh sáng không đều
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)
    
    cv2.imwrite("output2.jpg", thresh)

    # Morphological operation để kết nối ký tự
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
    dilation = cv2.dilate(thresh, kernel, iterations=1)
    
    cv2.imwrite("output3.jpg", dilation)

    # Tìm contour của các vùng chữ
    contours, _ = cv2.findContours(thresh, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

    boxes = []
    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)
        roi = thresh[y:y+h, x:x+w]  # Vùng quan tâm (ROI)
        black_ratio = np.sum(roi == 0) / (w * h)  # Tỷ lệ màu đen
        
        # Kiểm tra kích thước hợp lý và tỷ lệ màu đen
        if w > 150 and h > 120 and w > h:  # Tỷ lệ màu đen ít nhất 10%
            #print(f"Contour {x} {y} {w} {h}")
            boxes.append((x, y, x + w, y + h))

    return boxes, image

def preprocess_for_ocr(image):
    """Xử lý ảnh để cải thiện OCR."""
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Adaptive threshold để làm chữ nổi bật
    thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                   cv2.THRESH_BINARY, 11, 2)

    return thresh

def extract_text_and_highlight(image_path, boxes, image):
    """Dùng OCR để trích xuất text từ các vùng đã phát hiện."""
    detected_texts = []
    detected_txts = {}

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        cropped = image[y1:y2, x1:x2]  # Cắt vùng bounding box
        #processed = preprocess_for_ocr(cropped)  # Xử lý ảnh trước OCR

        # Chuyển sang PIL Image để sử dụng pytesseract
        pil_image = Image.fromarray(cropped)
        config = "--psm 6 -c tessedit_char_whitelist=ABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789.-,"
        #config = "--psm 6"
        # Sử dụng image_to_data để lấy thông tin chi tiết
        ocr_data = pytesseract.image_to_data(pil_image, config=config, output_type=pytesseract.Output.DICT)
        texts = [ocr_data['text'][ii].strip() for ii in range(len(ocr_data["text"])) if len(ocr_data['text'][ii]) > 1 and ocr_data['conf'][ii] > 1 and validate_decimal_format(ocr_data['text'][ii])]
        #print("(%s, %s, %s, %s) len: %s" % (x1, y1, x2, y2, "-".join(texts)) )
        if len(texts) > 0:
            final_txt = "-".join(texts).replace(',', '.')
            #print('detected_texts', detected_texts)
            if len(final_txt) > 10 and validate_license(final_txt):
                #detected_txts[final_txt] = 0
                #detected_texts.append((final_txt, (x1, y1, x2, y2), 100))
                print(final_txt)
                break
        #for j in range(len(ocr_data["text"])):
            #text = ocr_data["text"][j].strip()
            #confidence = int(ocr_data["conf"][j])
            #x, y, w, h = (ocr_data["left"][j], ocr_data["top"][j], ocr_data["width"][j], ocr_data["height"][j])
            
            #if text and confidence > 15:  # Lọc kết quả dựa trên độ tự tin
                #print(f"Detected Text {i}-{j}: '{text}' (Confidence: {confidence}) at ({x}, {y}, {x + w}, {y + h})")

    return detected_texts 

def validate_decimal_format(text):
    """Kiểm tra xem text có khớp định dạng ddd.dd hoặc chứa 3 chữ số."""
    cleaned_text = re.sub(r"\s+", "", text)  # Xóa xuống dòng và khoảng trắng
    pattern = r"\d{2}"  # Tìm 3 chữ số liên tiếp
    return re.search(pattern, cleaned_text) and len(text) >= 5

def validate_license(text):
    """Kiểm tra xem text có khớp định dạng ddd.dd hoặc chứa 3 chữ số."""
    cleaned_text = re.sub(r"\s+", "", text)  # Xóa xuống dòng và khoảng trắng
    pattern = r"\d{3}.\d{2}"  # Tìm 3 chữ số liên tiếp
    return re.search(pattern, cleaned_text)

def draw_boxes(image, detected_texts, output_path):
    """Vẽ khung quanh các vùng được phát hiện."""
    for text, (x1, y1, x2, y2), confidence in detected_texts:
        cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)  # Khung màu xanh lá
        label = f"{text} ({confidence}%)"
        cv2.putText(image, label, (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)

    # Lưu ảnh đã chỉnh sửa
    cv2.imwrite(output_path, image)
    #print(f"Saved output image with highlighted boxes to {output_path}")

def detect_plate(image_path, output_path):
    """Phát hiện và xử lý biển số xe."""
    boxes, image = detect_text_regions(image_path)
    #print(f"Detected {len(boxes)} text regions.")

    # Dùng OCR để kiểm tra vùng khớp
    detected_texts = extract_text_and_highlight(image_path, boxes, image)

    # Vẽ khung cho các vùng phù hợp
    draw_boxes(image, detected_texts, output_path)

    #if detected_texts:
        #print("Final Detected Texts:")
        #for text, _, confidence in detected_texts:
            #print(f" - '{text}' (Confidence: {confidence}%)")
    #else:
        #print("No valid regions detected.")

# Chạy chương trình
if __name__ == "__main__":
    args = parse_args()
    detect_plate(args.image, args.output)
