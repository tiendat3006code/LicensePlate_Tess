import argparse
import pytesseract
from PIL import Image, ImageEnhance

# Parse command line arguments
def parse_args():
    parser = argparse.ArgumentParser(description="OCR License Plate Recognition")
    parser.add_argument('-i', '--image', required=True, help="Path to the image file")
    return parser.parse_args()

# Preprocess the image
def preprocess_image(image_path):
    # Open the image using PIL
    image = Image.open(image_path)

    # Step 1: Convert to grayscale (this often works well for OCR)
    #grayscale_image = image.convert('L')
    crop_image = image.crop((50,50,600,500))

    # Step 2: Enhance contrast (optional, can help with text visibility)
    #enhancer = ImageEnhance.Contrast(grayscale_image)
    #enhanced_image = enhancer.enhance(2.0)  # Increase contrast by a factor of 2.0

    return crop_image

# OCR function
def ocr(image):
    # Hardcoded PSM and OEM values
    psm = 6   # Page segmentation mode for uniform block of text
    oem = 3   # OCR Engine Mode for both standard and LSTM OCR engine

    # Use the pytesseract OCR engine with the specified PSM and OEM
    custom_config = f'--psm {psm} --oem {oem}'
    text = pytesseract.image_to_string(image, config=custom_config)

    # Post-process text (optional improvements)
    # Correct common OCR errors, like misinterpreting 'I' as '1'
    text = text.replace('I', '1')  # Example: convert I to 1 (this is common in license plates)
    text = text.replace('l', '1')  # Example: convert lowercase L to 1
    text = text.strip()  # Remove any extra whitespace

    return text

# Main function to run OCR
def main():
    # Parse arguments
    args = parse_args()

    # Preprocess the image
    preprocessed_image = preprocess_image(args.image)

    # Perform OCR on the preprocessed image
    result = ocr(preprocessed_image)

    # Print the extracted text
    print("Extracted Text:")
    print(result)

if __name__ == "__main__":
    main()
