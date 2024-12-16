import cv2
import json
import os
import random
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor
import torch

# Segment Class
class Segment:
    # Define the constructor
    def __init__(self, exp=23, exp_i=1):
        self.exp = exp
        self.exp_i = exp_i
        self.output_dir = f"_output-{self.exp}-{self.exp_i}"
        self.sam_checkpoint = "sam_vit_h_4b8939.pth"
        self.model_type = "vit_h"
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.sam = sam_model_registry[self.model_type](checkpoint=self.sam_checkpoint)
        self.sam.to(device=self.device)
        self.mask_generator = SamAutomaticMaskGenerator(self.sam)

    def process_image(self, image):
        masks = self.mask_generator.generate(image)
        input_points = []
        input_boxes = []
        for mask in masks:
            bbox = mask['bbox']
            x, y, w, h = bbox
            # Calculate the center point of the bounding box
            x_center = x + w / 2
            y_center = y + h / 2
            input_points.append([[x_center, y_center]])
            input_boxes.append(bbox)
        input_labels = [1] * len(masks)

        return input_points, input_boxes, input_labels

if __name__ == "__main__":
    segmenter = Segment()
    image_name = 'NORMAL-9251-1.jpeg'
    image_path = os.path.join(os.path.dirname(__file__), 'data', 'oct-5', 'normal', image_name)
    image = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    input_points, input_boxes, input_labels = segmenter.process_image(image)
