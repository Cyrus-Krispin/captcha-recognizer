import cv2
import numpy as np
import os
from matplotlib import pyplot as plt

class CaptchaSegmenter:
    def __init__(self, image_path, output_folder="extracted_letter_images", min_area=20):
        self.image_path = image_path
        self.output_folder = output_folder
        self.min_area = min_area
        self.image = None
        self.processed_image = None
        self.contours = None
        self.letter_regions = []
        self.captcha_text = None
        self.average_character_width = 0

        # Create the output folder if it doesn't exist
        if not os.path.exists(self.output_folder):
            os.makedirs(self.output_folder)

    def load_image(self):
        """Loads the CAPTCHA image and converts it to grayscale."""
        self.image = cv2.imread(self.image_path)
        self.image = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
        self.image = clahe.apply(self.image)

    def load_captcha_text_from_file_name(self):
        """Extracts the CAPTCHA text from the image file name."""
        captcha_text = os.path.basename(self.image_path).split(".")[0]
        captcha_text = captcha_text.split("-")[0]
        self.captcha_text = captcha_text
        print(f"CAPTCHA text: {captcha_text}")

    def preprocess_image(self):
        """Applies adaptive thresholding and median blur for segmentation."""
        # Apply adaptive thresholding to create a binary inverted image
        im_bw_inverted = cv2.adaptiveThreshold(
            self.image, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )
        # Apply median blur to reduce noise
        im_bw_inverted = cv2.medianBlur(im_bw_inverted, 3)

        self.processed_image = im_bw_inverted

    def apply_morphology(self, kernel_size=(1, 1)):
        """Applies morphological closing to connect character parts."""
        kernel = np.ones(kernel_size, np.uint8)
        self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)

    def find_contours(self):
        """Finds contours in the processed image."""
        self.contours = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.contours[0] if len(self.contours) == 2 else self.contours[1]

    def adaptive_morphology_and_contours(self):
        """Dynamically apply different kernel sizes to obtain the best contour results."""
        for kernel_size in [(1, 1), (2, 2), (3, 3), (4, 4)]:
            self.apply_morphology(kernel_size=kernel_size)
            self.find_contours()

            character_widths = [cv2.boundingRect(c)[2] for c in self.contours if cv2.contourArea(c) > self.min_area]
            if len(character_widths) == len(self.captcha_text):
                print(f"Using kernel size: {kernel_size}")
                self.extract_character_regions()
                break

    def extract_character_regions(self):
        """Extract character regions and update letter regions."""
        character_widths = [cv2.boundingRect(c)[2] for c in self.contours if cv2.contourArea(c) > self.min_area]
        mean = np.mean(character_widths)
        std_dev = np.std(character_widths)
        threshold = mean + std_dev
        filtered_numbers = [num for num in character_widths if num <= threshold]
        self.average_character_width = np.median(filtered_numbers) if filtered_numbers else 0

        for c in self.contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                (x, y, w, h) = cv2.boundingRect(c)
                if w < 2 * self.average_character_width:
                    self.letter_regions.append((x, y, w, h))

    def segment_characters(self):
        """Sort the detected letter images based on the x coordinate and save each character as a single image."""
        self.adaptive_morphology_and_contours()

        if not self.letter_regions:
            return

        # Sort the letter regions based on the x coordinate
        self.letter_regions = sorted(self.letter_regions, key=lambda x: x[0])

        char_index = 0
        ROIs = []  # List to store ROIs for all characters

        # Process each character region
        for (x, y, w, h) in self.letter_regions:
            if w > 2 * self.average_character_width:
                num_segments = int(round(w / self.average_character_width))
                segment_width = w // num_segments

                for i in range(num_segments):
                    x_segment = x + i * segment_width
                    w_segment = segment_width if i < num_segments - 1 else (w - i * segment_width)
                    ROI = self.processed_image[y:y + h, x_segment:x_segment + w_segment]
                    try:
                        ROIs.append((ROI, self.captcha_text[char_index]))  # Store ROI and corresponding character text
                        char_index += 1
                    except:
                        print("Skipping image due to incorrect number of ROIs")
                        self.letter_regions = []
                        return
            else:
                ROI = self.processed_image[y:y + h, x:x + w]
                try:
                    ROIs.append((ROI, self.captcha_text[char_index]))  # Store ROI and corresponding character text
                    char_index += 1
                except:
                    print("Skipping image due to incorrect number of ROIs")
                    self.letter_regions = []
                    return

        # After collecting all ROIs, write them to disk
        if len(ROIs) != len(self.captcha_text):
            print("Skipping image due to incorrect number of ROIs")
            self.letter_regions = []  # Clear the letter regions
            return
        for roi, character in ROIs:
            output_dir = os.path.join(self.output_folder, character)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_number = len(os.listdir(output_dir)) + 1
            cv2.imwrite(os.path.join(output_dir, f"{image_number}.png"), roi)

        # Clear the letter regions after processing
        self.letter_regions = []

    def run_segmentation(self):
        """Runs the full segmentation process."""
        self.load_captcha_text_from_file_name()
        self.load_image()
        self.preprocess_image()
        self.segment_characters()

# Process all the files in the main folder
for image_path in os.listdir("main"):
    if image_path.endswith(".png"):
        print(f"Processing {image_path}")
        segmenter = CaptchaSegmenter(os.path.join("main", image_path))
        segmenter.run_segmentation()
