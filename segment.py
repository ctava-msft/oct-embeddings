import cv2
import numpy as np
import os
from scipy.ndimage import gaussian_filter
from scipy import signal

class Segment:
    # Define the constructor
    def __init__(self, exp=23, exp_i=1):
        self.exp = exp
        self.exp_i = exp_i
        self.output_dir = f"_output-{self.exp}-{self.exp_i}"
        os.makedirs(self.output_dir, exist_ok=True)
        self.segment_dir = f"{self.output_dir}/segments"
        os.makedirs(self.segment_dir, exist_ok=True)

    # Define the method to filter the signal
    def filter_signal(self, image):
        # High-pass filter specifications
        Fs = 1000        # Sample rate (Hz)
        fpass = 200      # Passband cutoff (Hz)
        fstop = 150      # Stopband cutoff (Hz)
        fc = 175         # Highpass cutoff frequency (Hz)
        deltadB = 60     # Minimum desired attenuation in stopband
        beta = 8.6       # Kaiser window beta parameter
        # Design FIR High-Pass Filter
        M, beta = signal.kaiserord(deltadB, (fpass - fstop) / (Fs / 2))
        if M % 2 == 0:
            M += 1  # Ensure M is odd
        b = signal.firwin(M, fc, window=('kaiser', beta), pass_zero=False, fs=Fs)
        return signal.lfilter(b, 1, image)

    # Define the method to obtain input coordinates
    def obtain_input_coordinates(self, image):

        # Ensure the image is in uint8 format
        if image.dtype != np.uint8:
            image = cv2.normalize(image, None, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

        output_image = image.copy()
        output_image = cv2.cvtColor(image, cv2.COLOR_GRAY2BGR)
        # Apply Gaussian smoothing
        smoothed_image = gaussian_filter(output_image, sigma=0.9).astype(np.uint8)

        t1=10
        t2=70
        edges = cv2.Canny(smoothed_image, threshold1=t1, threshold2=t2)
        # Apply dilation to connect fragmented edges
        kernel = np.ones((3, 3), np.uint8)
        edges = cv2.dilate(edges, kernel, iterations=1)
        # Invert edges for correct visualization
        edges = cv2.bitwise_not(edges)
        # Find contours from the edges
        #contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        #contours, _ = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)
        contours_tuple = cv2.findContours(edges, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)

        contours = contours_tuple[0]  # Extract the contours list
        # If contours is a tuple, convert it to a list
        if isinstance(contours, tuple):
            contours = list(contours)

        MIN_AREA = 1000
        contours = [cnt for cnt in contours if cv2.contourArea(cnt) > MIN_AREA]

        print(f"Number of contours detected: {len(contours)}")
        #print(f"Type of contours: {type(contours)}")
        #print(f"Sample contour: {contours[0] if len(contours) > 0 else 'No contours'}")
        # Ensure contours is a list of numpy arrays
        if not isinstance(contours, list):
            raise TypeError(f"Contours should be a list, but got {type(contours)}")
        for i, contour in enumerate(contours):
            if not isinstance(contour, np.ndarray):
                raise TypeError(f"Contour at index {i} should be a numpy array, but got {type(contour)}")

        # Generate input_points, input_boxes, and input_labels based on contours
        input_points = []
        input_boxes = []
        input_labels = []
        
        for idx, contour in enumerate(contours):
            # Calculate centroid for input_points
            M = cv2.moments(contour)
            if M["m00"] != 0:
                cX = int(M["m10"] / M["m00"])
                cY = int(M["m01"] / M["m00"])
                input_points.append([cX, cY])
            
            # Calculate bounding box for input_boxes
            x, y, w, h = cv2.boundingRect(contour)
            input_boxes.append([x, y, x + w, y + h])

            # Generate label for input_labels
            label = f"Layer_{idx}"
            input_labels.append(label)
        
        # Print or save the lists as needed
        print("input_points:", input_points)
        print("input_boxes:", input_boxes)
        print("input_labels:", input_labels)
        return input_points, input_boxes, input_labels

    def process_image(self, image):
        filtered_image = self.filter_signal(image)
        input_points, input_boxes, input_labels = self.obtain_input_coordinates(filtered_image)
        return input_points, input_boxes, input_labels

if __name__ == "__main__":
    segmenter = Segment()
    image_name = 'NORMAL-9251-1.jpeg'
    image_path = os.path.join(os.path.dirname(__file__), 'data', 'oct-5', 'normal', image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_points, input_boxes, input_labels = segmenter.process_image(image)
