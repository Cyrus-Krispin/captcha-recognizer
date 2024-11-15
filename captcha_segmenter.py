import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class CaptchaSegmenter:

    skipped = 0

    def __init__(self, image_path, output_folder="extracted_letter_images", min_area=20):
        self.image_path = image_path
        self.output_folder = output_folder
        self.image = cv2.imread(image_path)
        self.cropped_image = None
        self.detected_letters = []
        self.captcha_text = None
        self.line_coordinates = []
        
    def load_captcha_text_from_file_name(self):
        """Extracts the CAPTCHA text from the image file name."""
        captcha_text = os.path.basename(self.image_path).split(".")[0]
        captcha_text = captcha_text.split("-")[0]
        self.captcha_text = captcha_text
        print(f"CAPTCHA text: {captcha_text}")
        
        
    def run_segmentation(self):
        grayscale = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        hist_adjusted = cv2.equalizeHist(grayscale)
        adaptive_thresh = cv2.adaptiveThreshold(hist_adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)

        median_filtered = cv2.medianBlur(adaptive_thresh, 3)


        morphology_kernel = np.ones((3, 3), np.uint8)
        opened_img = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, morphology_kernel, iterations=1)

        opened_img_contours, _ = cv2.findContours(opened_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        if not opened_img_contours:
            print("No contours found in the image.")
            return

        all_points = np.vstack(opened_img_contours).squeeze()
        all_x, all_y = all_points[:, 0], all_points[:, 1]

        pad = 0
        height, width = opened_img.shape
        x_min, x_max = max(0, min(all_x) - pad), min(width, max(all_x) + pad)
        y_min, y_max = max(0, min(all_y) - pad), min(height, max(all_y) + pad)

        x_max_crop = x_max

        captcha_area = median_filtered[y_min:y_max, x_min:x_max]
        captcha_area_colored = self.image[y_min:y_max, x_min:x_max]
            
        captcha_contours, _ = cv2.findContours(captcha_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        bounding_boxes_coords = sorted([cv2.boundingRect(contour) for contour in captcha_contours], key=lambda x: x[0])

        grouped_boxes = []
        current_group = bounding_boxes_coords[0]

        for box in bounding_boxes_coords[1:]:
            x, y, w, h = box
            prev_x, prev_y, prev_w, prev_h = current_group
            overlap_width = min(prev_x + prev_w, x + w) - max(prev_x, x)

            if overlap_width > 0 and (overlap_width >= 0.1 * w or overlap_width >= 0.1 * prev_w):
                current_group = (
                    min(prev_x, x),
                    min(prev_y, y),
                    max(prev_x + prev_w, x + w) - min(prev_x, x),
                    max(prev_y + prev_h, y + h) - min(prev_y, y)
                )
            else:
                grouped_boxes.append(current_group)
                current_group = box

        grouped_boxes.append(current_group)
        grouped_boxes.sort(key=lambda x: x[0])

        letters = []

        for i in range(len(grouped_boxes)):
            x_box, y_box, w_box, h_box = grouped_boxes[i]
            segment = captcha_area_colored[:, x_box:min(x_box + w_box, x_max_crop)]

            if segment is None or segment.size == 0 or segment.shape[1] == 0:
                continue

            hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
            hue, saturation, value = cv2.split(hsv_segment)

            segment_mask = captcha_area[:, x_box:x_box + w_box]
            segment_mask_resized = cv2.resize(segment_mask, (hue.shape[1], hue.shape[0])) 
            segment_mask_binary = cv2.threshold(segment_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]

            hist = cv2.calcHist([hue], [0], segment_mask_binary, [180], [0, 180]).flatten()
            peak_threshold = 0.3 * max(hist)
            peaks = np.sort(np.where(hist > peak_threshold)[0])
            peaks = peaks[peaks > 0]

            combined_peaks = [peaks[0]] if len(peaks) > 0 else []

            for peak in peaks[1:]:
                if abs(peak - combined_peaks[-1]) >= 2:
                    combined_peaks.append(peak)

            peaks = np.array(combined_peaks)

            segment_contours = []

            for peak in peaks:
                lower_bound = np.array([max(peak - 2, 0)], dtype=np.uint8)
                upper_bound = np.array([min(peak + 2, 179)], dtype=np.uint8)

                peak_mask = cv2.inRange(hue, lower_bound, upper_bound)
                filtered_segment = cv2.bitwise_and(hsv_segment, hsv_segment, mask=peak_mask)

                gray = cv2.cvtColor(filtered_segment, cv2.COLOR_BGR2GRAY)
                eq = cv2.equalizeHist(gray)
                
                _, bin = cv2.threshold(eq, 1, 255, cv2.THRESH_BINARY)
                contours, _ = cv2.findContours(bin, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Get x, y coordinates of letters
                all_x, all_y = [], []
                for contour in contours:
                    for point in contour:
                        all_x.append(point[0][0])
                        all_y.append(point[0][1])

                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)

                if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
                    continue
                
                padding = 2
                padded_bin = cv2.copyMakeBorder(
                    bin[:, x_min:x_max],  # The sliced binary image
                    padding,              # Padding for the top
                    padding,              # Padding for the bottom
                    padding,              # Padding for the left
                    padding,              # Padding for the right
                    cv2.BORDER_CONSTANT,  # Border type
                    value=0               # Padding value
                )
        
                letters.append((padded_bin, (x_box + x_min, y_box + y_min, w_box, h_box)))


        letters.sort(key=lambda letter: letter[1][0])
        self.detected_letters = [letter[0] for letter in letters]
            
    def image_resize(self, image, width, height, inter = cv2.INTER_AREA):
        # initialize the dimensions of the image to be resized and
        # grab the image size
        dim = None
        (h, w) = image.shape[:2]

        # if both the width and height are None, then return the
        # original image
        if width is None and height is None:
            return image

        # check to see if the width is None
        if width is None:
            # calculate the ratio of the height and construct the
            # dimensions
            r = height / float(h)
            dim = (int(w * r), height)

        # otherwise, the height is None
        else:
            # calculate the ratio of the width and construct the
            # dimensions
            r = width / float(w)
            dim = (width, int(h * r))

        # resize the image
        resized = cv2.resize(image, dim, interpolation = inter)

        # return the resized image
        return resized
        
    
    def store_extracted_letter_images(self):
        char_index = 0
        print(len(self.detected_letters), len(self.captcha_text))
        if len(self.detected_letters) != len(self.captcha_text):
            print("Detected letters do not match the CAPTCHA text.")
            CaptchaSegmenter.skipped += 1
            mismatch_dir = "mismatched_images"
            if not os.path.exists(mismatch_dir):
                os.makedirs(mismatch_dir)
            image_name = os.path.basename(self.image_path)
            cv2.imwrite(os.path.join(mismatch_dir, image_name), self.image)
            return

        for i in range(len(self.detected_letters)):
            letter_image = self.image_resize(self.detected_letters[i], 40, 40)
            output_dir = os.path.join(self.output_folder, self.captcha_text[char_index])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_number = len(os.listdir(output_dir)) + 1
            cv2.imwrite(os.path.join(output_dir, f"{image_number}.png"), letter_image)
            char_index += 1

            
for image_path in os.listdir("main"):
    if image_path.endswith(".png"):
        print(f"Processing {image_path}")
        segmenter = CaptchaSegmenter(os.path.join("main", image_path))
        segmenter.load_captcha_text_from_file_name()
        segmenter.run_segmentation()
        segmenter.store_extracted_letter_images()


print(f"Skipped {CaptchaSegmenter.skipped} character images.")
