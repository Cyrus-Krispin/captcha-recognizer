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
            self.image, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 15, 2
        )

        # Apply median blur to reduce noise
        im_bw_inverted = cv2.medianBlur(im_bw_inverted, 3)

        self.processed_image = im_bw_inverted

    def apply_morphology(self, kernel_size=(1, 1)):
        """Applies morphological closing to connect character parts."""
        kernel = np.ones(kernel_size, np.uint8)  # Adjust kernel size as needed
        self.processed_image = cv2.morphologyEx(self.processed_image, cv2.MORPH_CLOSE, kernel)
        

    def find_contours(self):
        """Finds contours in the processed image."""
        self.contours = cv2.findContours(self.processed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        self.contours = self.contours[0] if len(self.contours) == 2 else self.contours[1]

    def get_character_regions_with_kernel_1(self):
        """Extracts character regions from the contours."""
        self.apply_morphology(kernel_size=(1, 1))
        self.find_contours()
        
        character_widths = []
        # Collect character widths
        for c in self.contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                (x, y, w, h) = cv2.boundingRect(c)
                character_widths.append(w)

        mean = np.mean(character_widths)
        std_dev = np.std(character_widths)
        threshold = mean + std_dev 
        filtered_numbers = [num for num in character_widths if num <= threshold]

        self.average_character_width = np.max(filtered_numbers)
        
        for c in self.contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                (x, y, w, h) = cv2.boundingRect(c)
                self.letter_regions.append((x, y, w, h))
        
        if len(self.letter_regions) != len(self.captcha_text):
            # skip this image if the number of detected regions is greater than the number of character
            self.letter_regions = []
        
        # if len(self.letter_regions) < len(self.captcha_text):
            # have to change kernel size maybe
                

        print(f"Detected {len(self.letter_regions)} character regions")
    
    def get_character_regions_with_kernel_2(self):
        """Extracts character regions from the contours."""
        self.apply_morphology(kernel_size=(2, 2))
        self.find_contours()
        print("Number of contours:", len(self.contours))
        for c in self.contours:
            area = cv2.contourArea(c)
            if area > self.min_area:
                (x, y, w, h) = cv2.boundingRect(c)
                
                    
                self.letter_regions.append((x, y, w, h))
        
        if len(self.letter_regions) != len(self.captcha_text):
            # skip this image if the number of detected regions is greater than the number of character
            self.letter_regions = []
        
        # if len(self.letter_regions) < len(self.captcha_text):
            # have to change kernel size maybe
                

        print(f"Detected {len(self.letter_regions)} character regions")
    
    
    def segment_characters(self):
        """sort the detected letter images based on the x coordinate and save each character as a single image"""
        self.get_character_regions_with_kernel_2()
        if not self.letter_regions:
            self.get_character_regions_with_kernel_1()
        if not self.letter_regions:
            return
        self.letter_regions = sorted(self.letter_regions, key=lambda x: x[0])
        char_index = 0
        for (x, y, w, h) in self.letter_regions:
            # Draw bounding box around each character
            cv2.rectangle(self.image, (x, y), (x + w, y + h), (36, 255, 12), 2)
            
            # Extract and save each character as a Region of Interest (ROI)
            ROI = self.processed_image[y:y + h, x:x + w]
            
            # Save each segmented character
            output_dir = os.path.join(self.output_folder, self.captcha_text[char_index])
            
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            
            # if it exits, count the number of files in the directory and increment the image number
            image_number = len(os.listdir(output_dir)) + 1
                
            cv2.imwrite(os.path.join(output_dir, f"{image_number}.png"), ROI)
            char_index += 1
        
        self.letter_regions = []

    def run_segmentation(self):
        """Runs the full segmentation process."""
        # print("Captcha text:", self.load_captcha_text_from_file_name())
        self.load_captcha_text_from_file_name()
        # print("Loading image...")
        self.load_image()
        # print("Preprocessing image...")
        self.preprocess_image()
        # print("Applying morphological operations...")
        # self.apply_morphology()
        # print("Finding contours...")
        # self.find_contours()
        # print("Segmenting characters...")
        self.segment_characters()
        # print("Segmentation completed!")


image_path = "train/a70ond9c-0.png" 
segmenter = CaptchaSegmenter(image_path)
segmenter.run_segmentation()

# for all the files in the train folder
# for image_path in os.listdir("train"):
#     if image_path.endswith(".png"):
#         print(f"Processing {image_path}")
#         segmenter = CaptchaSegmenter(os.path.join("train", image_path))
#         segmenter.run_segmentation()
