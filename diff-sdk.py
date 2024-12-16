import torch
from segment_anything import sam_model_registry
from PIL import Image
import numpy as np
from torchvision import transforms

# Load the model
sam_checkpoint="./sam_vit_h_4b8939.pth"
model_type = "vit_h"
sam = sam_model_registry[model_type](checkpoint=sam_checkpoint)
device = "cuda" if torch.cuda.is_available() else "cpu"
sam.to(device=device)

# Get the embeddings
def get_embeddings(image_paths):
    transform = transforms.ToTensor()
    images = [transform(Image.open(image_path).convert("RGB")) for image_path in image_paths]
    preprocessed_images = [sam.preprocess(image).to(device) for image in images]
    #images = [sam.preprocess(Image.open(image_path)).unsqueeze(0).to(device) for image_path in image_paths]
    with torch.no_grad():
        embeddings = sam.encode_image(preprocessed_images)
    return embeddings.cpu().numpy()

# Compute the difference
def compute_diff(x, y):
    # Compute the dot product of x with its transpose
    xx = np.dot(x, x.T)
    # Compute the dot product of y with its transpose
    yy = np.dot(y, y.T)
    # Compute the dot product of x with y's transpose
    xy = np.dot(x, y.T)
    # Extract the diagonal elements of xx
    rx = np.diag(xx)
    # Extract the diagonal elements of yy
    ry = np.diag(yy)
    # Calculate the kernel matrix for x
    k = np.exp(-0.5 * (rx[:, None] + rx[None, :] - 2 * xx))
    # Calculate the kernel matrix for y
    l = np.exp(-0.5 * (ry[:, None] + ry[None, :] - 2 * yy))
    # Calculate the cross-kernel matrix between x and y
    m = np.exp(-0.5 * (rx[:, None] + ry[None, :] - 2 * xy))
    # Compute and return the mean difference
    return np.mean(k) + np.mean(l) - 2 * np.mean(m)

# Calculate the difference
def calculate_diff(image_paths1, image_paths2):
    embeddings1 = get_embeddings(image_paths1)
    embeddings2 = get_embeddings(image_paths2)
    return compute_diff(embeddings1, embeddings2)

# Execute the function
image_paths1 = ["./data/samples/image.png"]
image_paths2 = ["./data/samples/image.png"]
value = calculate_diff(image_paths1, image_paths2)
print(f"Diff: {value}")