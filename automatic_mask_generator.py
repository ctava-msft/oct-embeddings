import json
import numpy as np
import os
import random
import torch
import matplotlib.pyplot as plt
import cv2
import sys
sys.path.append("..")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator, SamPredictor

def show_anns(anns):
    if len(anns) == 0:
        return
    sorted_anns = sorted(anns, key=(lambda x: x['area']), reverse=True)
    ax = plt.gca()
    ax.set_autoscale_on(False)

    img = np.ones((sorted_anns[0]['segmentation'].shape[0], sorted_anns[0]['segmentation'].shape[1], 4))
    img[:,:,3] = 0
    for ann in sorted_anns:
        m = ann['segmentation']
        color_mask = np.concatenate([np.random.random(3), [0.35]])
        img[m] = color_mask
    ax.imshow(img)

image_paths1 = "./data/samples/normal.jpg"
image = cv2.imread(image_paths1)
if image is None:
    print(f"Failed to load image from {image_paths1}")
    sys.exit(1)
image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# plt.figure(figsize=(20,20))
# plt.imshow(image)
# plt.axis('off')
# plt.show()

sam_checkpoint = "sam_vit_h_4b8939.pth"
model_type = "vit_h"

device = "cuda" if torch.cuda.is_available() else "cpu"

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
sam.to(device=device)

mask_generator = SamAutomaticMaskGenerator(sam)

# Mask generation returns a list over masks, where each mask is a dictionary containing various data about the mask. These keys are:
# * `segmentation` : the mask
# * `area` : the area of the mask in pixels
# * `bbox` : the boundary box of the mask in XYWH format
# * `predicted_iou` : the model's own prediction for the quality of the mask
# * `point_coords` : the sampled input point that generated this mask
# * `stability_score` : an additional measure of mask quality
# * `crop_box` : the crop of the image used to generate this mask in XYWH format

masks = mask_generator.generate(image)

plt.figure(figsize=(20,20))
plt.imshow(image)
show_anns(masks)
plt.axis('off')
plt.show()

# Convert segmentation to list after visualization
for mask in masks:
    mask['segmentation'] = mask['segmentation'].tolist()

print(len(masks))
print(masks[0].keys())

output_path = os.path.join(os.path.dirname(__file__), 'data', 'samples', 'masks.json')
random_suffix = random.randint(1000, 9999)
output_path = output_path.replace(".json", f"_{random_suffix}.json")

with open(output_path, 'w') as f:
    json.dump(masks, f, indent=4)

# mask_generator_2 = SamAutomaticMaskGenerator(
#     model=sam,
#     points_per_side=32,
#     pred_iou_thresh=0.86,
#     stability_score_thresh=0.92,
#     crop_n_layers=1,
#     crop_n_points_downscale_factor=2,
#     min_mask_region_area=100,  # Requires open-cv to run post-processing
# )

# masks2 = mask_generator_2.generate(image)

# len(masks2)


# plt.figure(figsize=(20,20))
# plt.imshow(image)
# show_anns(masks2)
# plt.axis('off')
# plt.show()