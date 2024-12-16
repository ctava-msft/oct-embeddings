import matplotlib.pyplot as plt
import numpy as np
import os
import torch
from PIL import Image
import json

# If you cloned Segment-Anything repo, make sure its path is accessible
# For example:
# sys.path.append("path/to/segment-anything")
from segment_anything import sam_model_registry, SamAutomaticMaskGenerator

# Load the image

image_name = 'image.png'
#image_name = 'oct-id-105.jpg'
#image_name = 'kaggle-NORMAL-3099713-1.jpg'
#image_name = 'oct-500-3-10301-1.bmp'
image_path = os.path.join(os.path.dirname(__file__), 'data', 'samples', image_name)
image = Image.open(image_path).convert("RGB")
image_np = np.array(image)

# Initialize the SAM model
# You need to download a SAM checkpoint from the model zoo at:
# https://github.com/facebookresearch/segment-anything#model-checkpoints
sam_checkpoint = "sam_vit_h_4b8939.pth"  # Replace with the path to your SAM checkpoint
model_type = "vit_h"  # or vit_l, vit_b depending on which checkpoint you use

sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)  # Removed .cuda()
mask_generator = SamAutomaticMaskGenerator(sam)   # Removed device argument

# Generate masks automatically (no prompts)
masks = mask_generator.generate(image_np)

# Prepare input_points, input_boxes, input_labels
# input_boxes will be derived from mask bounding boxes
# We can choose one point per bounding box and label them as foreground (e.g. label=1)
input_points = []
input_labels = []
input_boxes = []

for m in masks:
    # Each mask dictionary has 'bbox' in [x_min, y_min, width, height] format
    bbox = m["bbox"]
    x0, y0, w, h = bbox
    x1 = x0 + w
    y1 = y0 + h
    # Convert to [x0, y0, x1, y1] format for consistency
    box_coords = [x0, y0, x1, y1]
    input_boxes.append(box_coords)

    # Pick a point in the center of the box as an input point
    cx = int((x0 + x1) / 2)
    cy = int((y0 + y1) / 2)
    input_points.append([cx, cy])
    # Label this point as foreground
    input_labels.append(1)

# Store the variables in a JSON file
data = {
    "input_points": input_points,
    "input_boxes": input_boxes,
    "input_labels": input_labels
}
output_path = os.path.join(os.path.dirname(__file__), 'data', 'output.json')
with open(output_path, 'w') as f:
    json.dump(data, f, indent=4)

# Optional: If you do not need prompt points/labels for this demonstration,
# you could leave input_points and input_labels empty. For demonstration, we have them:
# input_points = []
# input_labels = []

# Visualization
fig, ax = plt.subplots(figsize=(10, 10))
ax.imshow(image_np)

# Function to show box
def show_box(box, ax):
    x0, y0 = box[0], box[1]
    w, h = box[2] - box[0], box[3] - box[1]
    ax.add_patch(
        plt.Rectangle((x0, y0), w, h, edgecolor="green", facecolor=(0, 0, 0, 0), lw=2)
    )

# Function to show mask
def show_mask(mask, ax, random_color=False):
    if not isinstance(mask, np.ndarray):
        mask = np.array(mask)
        mask = mask > 0
    if random_color:
        color = np.concatenate([np.random.random(3), np.array([0.6])], axis=0)
    else:
        color = np.array([30 / 255, 144 / 255, 1.0, 0.6])
    h, w = mask.shape[-2:]
    mask_image = mask.reshape(h, w, 1) * color.reshape(1, 1, -1)
    ax.imshow(mask_image)

# Display each mask and bounding box
for m in masks:
    show_mask(m["segmentation"], ax)
for box in input_boxes:
    show_box(box, ax)

# Plot input points if you want to visualize them
for pt, lbl in zip(input_points, input_labels):
    color = "red" if lbl == 1 else "blue"
    ax.plot(pt[0], pt[1], marker="o", markersize=5, color=color)

ax.axis("off")
plt.title("Segment Anything Model - Automatic Masks and Boxes")
plt.show()

# The variables are now:
# input_points: list of [x, y] coordinates for each object
# input_boxes: list of bounding boxes [x0, y0, x1, y1] for each object
# input_labels: list of labels corresponding to each point
