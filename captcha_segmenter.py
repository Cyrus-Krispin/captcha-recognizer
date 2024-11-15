import cv2
import numpy as np
from matplotlib import pyplot as plt
import os


class CaptchaSegmenter:
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
        gray = cv2.cvtColor(self.image, cv2.COLOR_BGR2GRAY)
        adjusted = cv2.equalizeHist(gray)

        # Apply adaptive thresholding to enhance character edges
        adaptive_filtered = cv2.adaptiveThreshold(adjusted, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                                cv2.THRESH_BINARY_INV, 11, 2)
       

        # Apply median filtering to remove lines and smooth the CAPTCHA image
        # Adjust the kernel size (e.g., 3, 5, or 7) based on the thickness of lines to be removed
        median_filtered = cv2.medianBlur(adaptive_filtered, 3)
      

        # Apply a minimal closing operation
        kernel = np.ones((3, 3), np.uint8)  # Small kernel for minimal effect
        closed_image = cv2.morphologyEx(median_filtered, cv2.MORPH_OPEN, kernel, iterations=1)
       

        # Find contours to detect the main CAPTCHA area
        contours, _ = cv2.findContours(closed_image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        final = []
        # Determine the bounding box that covers all contours
        if not contours:
            print("No contours found in the image.")
            return

        # Get all x and y coordinates from all contours
        all_x = []
        all_y = []
        for contour in contours:
            for point in contour:
                all_x.append(point[0][0])
                all_y.append(point[0][1])

        # Find the min and max of x and y coordinates to form a bounding box
        x_min, x_max = min(all_x), max(all_x)
        y_min, y_max = min(all_y), max(all_y)

        pad = 10

        height, width = closed_image.shape
        x_min = max(0, x_min - pad)
        x_max = min(width, x_max + pad)
        x_max_crop = x_max
        y_min = max(0, y_min - pad)
        y_max = min(height, y_max + pad)

        # Crop the main CAPTCHA area based on the bounding box
        captcha_area = median_filtered[y_min:y_max, x_min:x_max]
       
            
        # Find contours in the cropped area to analyze letter positions
        contours, _ = cv2.findContours(captcha_area, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        ###
        contour_visualization = cv2.cvtColor(captcha_area, cv2.COLOR_GRAY2BGR)

        # Draw each contour with a unique color for distinction
        for i, contour in enumerate(contours):
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.drawContours(contour_visualization, [contour], -1, color, 2)

        # Display the image with contours visualized
       
        # Get bounding boxes for each contour
        bounding_boxes = [cv2.boundingRect(contour) for contour in contours]
        bounding_boxes = sorted(bounding_boxes, key=lambda x: x[0])  # Sort by x-coordinate

        # Group bounding boxes based on 70% overlap
        grouped_boxes = []
        current_group = bounding_boxes[0]


        for i in range(1, len(bounding_boxes)):
            
            x, y, w, h = bounding_boxes[i]
            prev_x, prev_y, prev_w, prev_h = current_group
            
            # Calculate overlap width
            overlap_width = min(prev_x + prev_w, x + w) - max(prev_x, x)
            
            # Check for 70% overlap of either bounding box
            if overlap_width > 0 and (overlap_width >= 0.6 * w or overlap_width >= 0.6 * prev_w):
                # Merge bounding boxes by extending current group
                x_min_1 = min(prev_x, x)
                x_max_1 = max(prev_x + prev_w, x + w)
                y_min_1 = min(prev_y, y)
                y_max_1 = max(prev_y + prev_h, y + h)
                current_group = (x_min_1, y_min_1, x_max_1 - x_min_1, y_max_1 - y_min_1)
            else:
                # No significant overlap; add the current group to grouped_boxes and start a new group
                grouped_boxes.append(current_group)
                current_group = bounding_boxes[i]

        # Append the last group
        grouped_boxes.append(current_group)

        # Visualize grouped bounding boxes
        grouped_visualization = cv2.cvtColor(captcha_area, cv2.COLOR_GRAY2BGR)
        for group in grouped_boxes:
            x, y, w, h = group
            color = (np.random.randint(0, 255), np.random.randint(0, 255), np.random.randint(0, 255))
            cv2.rectangle(grouped_visualization, (x, y), (x + w, y + h), color, 2)

        # Display the image with grouped bounding boxes visualized
     
        # Create a copy of the captcha_area to draw vertical lines on
        segmented_image = cv2.cvtColor(captcha_area, cv2.COLOR_GRAY2BGR)

        # Draw vertical lines between each grouped bounding box
        for i in range(1, len(grouped_boxes)):
            # Calculate x-position between grouped bounding boxes
            prev_x = grouped_boxes[i - 1][0] + grouped_boxes[i - 1][2]
            curr_x = grouped_boxes[i][0]
            line_x = (prev_x + curr_x) // 2

            # Draw the vertical line at this position
            cv2.line(segmented_image, (line_x, 0), (line_x, captcha_area.shape[0]), (255, 0, 0), 1)

        # Show the segmented image with vertical lines
        
        # import cv2
        # import numpy as np
        # from matplotlib import pyplot as plt

        # # Assuming `image` is the original color CAPTCHA image, `captcha_area` is the binary mask, and `grouped_boxes` contains bounding boxes

        # Convert the original cropped area (color) to HSV for color segmentation
        crop = self.image[y_min:y_max, x_min:x_max]  # Cropped color image for color analysis
        segmented_image = cv2.cvtColor(captcha_area, cv2.COLOR_GRAY2BGR)

        # sort the grouped boxes by x-coordinate
        grouped_boxes = sorted(grouped_boxes, key=lambda x: x[0])

        letters = []

        print("grouped_boxes: ", len(grouped_boxes))

        # Loop through each grouped box to analyze color transitions
        for i in range(len(grouped_boxes)):
            # Get the coordinates of the current bounding box
            x, y, w, h = grouped_boxes[i]
            segment = crop[:, x:min(x + w, x_max_crop)]  # Crop each section based on bounding box positions
            if segment.shape[1] == 0 or segment.size == 0:
                continue
            
            if segment is not None and segment.size > 0:
               
                # Convert the segment to HSV and extract channels
                hsv_segment = cv2.cvtColor(segment, cv2.COLOR_BGR2HSV)
            else:
                print(f"Skipping empty segment {i}")
                continue
            hue_channel = hsv_segment[:, :, 0]
            saturation_channel = hsv_segment[:, :, 1]
            value_channel = hsv_segment[:, :, 2]

            # Create a mask to ignore background
            segment_mask = captcha_area[:, x:x + w]  # Crop the mask to match the segment dimensions
            segment_mask_resized = cv2.resize(segment_mask, (hue_channel.shape[1], hue_channel.shape[0]))  # Ensure size match
            segment_mask_binary = cv2.threshold(segment_mask_resized, 127, 255, cv2.THRESH_BINARY)[1]  # Ensure binary mask

            # Calculate the hue histogram only on the foreground pixels
            hist = cv2.calcHist([hue_channel], [0], segment_mask_binary, [180], [0, 180])
            hist = hist.flatten()

            # Plot the histogram
            

            # Identify significant peaks in the histogram that indicate color regions
            peak_threshold = 0.20 * max(hist)  # Define a threshold for peak detection
            peaks = np.where(hist > peak_threshold)[0]
            # check for very close peaks and ignore them
            
            peaks = peaks[peaks > 0]
            peaks.sort()
            
            combined_peaks = []

            if len(peaks) > 0:
                current_peak = peaks[0]
                for i in range(1, len(peaks)):
                    if abs(peaks[i] - current_peak) < 5:
                        continue
                    else:
                        combined_peaks.append(current_peak)
                        current_peak = peaks[i]
                combined_peaks.append(current_peak)
            peaks = np.array(combined_peaks)
            
        
                

            print("Detected Peaks:", peaks)

            # Step 1: Filter out pixels not close to any peak or that are too close to white
            filtered_segment = np.zeros_like(hsv_segment)

            # Apply peak-based filtering to keep only relevant hues
            
            segment_contours = []
            
            # Loop over the detected hue peaks
            for peak in peaks:
                # Define a range around each peak (±20 degrees)
                lower_bound = np.array([max(peak - 8, 0)], dtype=np.uint8)
                upper_bound = np.array([min(peak + 8, 179)], dtype=np.uint8)
                
                # Create masks for hues within the peak range and remove near-white pixels
                peak_mask = cv2.inRange(hue_channel, lower_bound, upper_bound)
                kk = cv2.bitwise_and(hsv_segment, hsv_segment, mask=peak_mask)

                gr = cv2.cvtColor(kk, cv2.COLOR_BGR2GRAY)
                gg = cv2.equalizeHist(gr)
                af = cv2.adaptiveThreshold(gg, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY_INV, 11, 2)
                

                contours, _ = cv2.findContours(af, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

                # Get all x and y coordinates from all contours
                all_x = []
                all_y = []
                for contour in contours:
                    for point in contour:
                        all_x.append(point[0][0])
                        all_y.append(point[0][1])

                # Find the min and max of x and y coordinates to form a bounding box
                x_min, x_max = min(all_x), max(all_x)
                y_min, y_max = min(all_y), max(all_y)

                if x_min < 0 or x_max < 0 or y_min < 0 or y_max < 0:
                    continue

                box = (x + x_min, y + y_min, x_max - x_min, y_max - y_min)
                final.append(box)

                img = af[:, x_min: x_max]
               
                # Plot the colored image after using the filter peak
                if x_min == x_max:
                    continue
                colored_img = kk[:, x_min: x_max]
                

                # Plot the grayscale image after using the filter peak
                gray_img = cv2.cvtColor(colored_img, cv2.COLOR_BGR2GRAY)
                
                # Convert the grayscale image to binary by thresholding
                _, bw_img = cv2.threshold(gray_img, 1, 255, cv2.THRESH_BINARY)

                # Ensure the image is binary (0 or 255)
                bw_img[bw_img > 0] = 255

    
                letters.append((bw_img, (x + x_min, y + y_min, w, h)))
        letters = sorted(letters, key=lambda x: x[1][0])
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
            return
        # for (x, y, w, h) in self.detected_letters:
        #     print(x, y, w, h)
            
        #     letter_image = self.cropped_image[:, x:x+w]
        #     output_dir = os.path.join(self.output_folder, self.captcha_text[char_index])
        #     if not os.path.exists(output_dir):
        #         os.makedirs(output_dir)
        #     image_number = len(os.listdir(output_dir)) + 1
        #     cv2.imwrite(os.path.join(output_dir, f"{image_number}.png"), letter_image)
        #     char_index += 1
        

        for i in range(len(self.detected_letters)):
            letter_image = self.image_resize(self.detected_letters[i], 40, 40)
            output_dir = os.path.join(self.output_folder, self.captcha_text[char_index])
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            image_number = len(os.listdir(output_dir)) + 1
            cv2.imwrite(os.path.join(output_dir, f"{image_number}.png"), letter_image)
            char_index += 1
            
    
            
            
        


        # Display the final segmented image with color-based vertical lines at cluster boundaries
        




# image_path = "main/0hhit-0.png" 
# segmenter = CaptchaSegmenter(image_path)
# segmenter.load_captcha_text_from_file_name()
# segmenter.run_segmentation()
# segmenter.store_extracted_letter_images()

for image_path in os.listdir("main"):
    if image_path.endswith(".png"):
        print(f"Processing {image_path}")
        segmenter = CaptchaSegmenter(os.path.join("main", image_path))
        segmenter.load_captcha_text_from_file_name()
        segmenter.run_segmentation()
        segmenter.store_extracted_letter_images()
