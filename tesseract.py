import os
import cv2
import pytesseract
import shutil

def check_labels(input_folder, output_folder):
    # Iterate over each image in the dataset
    for image_file in os.listdir(input_folder):
        if not image_file.endswith(".png"):
            continue
        image_path = os.path.join(input_folder, image_file)
        # Load image and run OCR
        img = cv2.imread(image_path)
        gry = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        (h, w) = gry.shape[:2]
        # gry = cv2.resize(gry, (w*2, h*2))
        # cls = cv2.morphologyEx(gry, cv2.MORPH_CLOSE, None)
        # thr = cv2.threshold(cls, 0, 255, cv2.THRESH_BINARY | cv2.THRESH_OTSU)[1]
        custom_config = r'--oem 3 --psm 7 -c tessedit_char_whitelist=abcdefghijklmnopqrstuvwxyzABCDEFGHIJKLMNOPQRSTUVWXYZ0123456789'
        ocr_text = pytesseract.image_to_string(gry, config=custom_config).strip().lower()
        ocr_text = ocr_text.replace(" ", "").replace("\n", "")
        
        # Extract expected label from filename
        expected_text = image_file.split("-")[0]
        
        # print(ocr_text)
        # Compare OCR result with the expected label
        if len(ocr_text) != len(expected_text):
            print(f"Potential mismatch in file {image_file}: Expected '{expected_text}', got '{ocr_text}'")
            # Optionally save mismatches for manual inspection
            # Copy the file from train folder to output_folder
            shutil.copy2(os.path.abspath(train_folder) + f'\\{expected_text}-0.png', os.path.abspath(output_folder))
            # cv2.imwrite(os.path.join(output_folder, f'{expected_text}-0.png'), gry)

train_folder = "train"
input_folder = "error_images"
output_folder = "cleaned_images"
os.makedirs(output_folder, exist_ok=True)
check_labels(input_folder, output_folder)
