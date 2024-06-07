import os
import numpy as np
import torch.nn as nn
import torchvision
from PIL import Image
from collections import defaultdict
import random

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def save_image_as_1_224_224(image_tensor, output_image_path):
    # Convert the image tensor to a PIL image
    img = torchvision.transforms.ToPILImage()(image_tensor)

    # Resize the image to (224, 224)
    img_resized = img.resize((224, 224))

    # Convert the image to a NumPy array
    img_array = np.array(img_resized)

    # If the image has 3 channels (RGB), convert it to a single channel grayscale image
    if img_array.ndim == 3:
        img_array = np.mean(img_array, axis=2)

    # Reshape the array to (1, 224, 224)
    img_array_reshaped = img_array.reshape((1, 224, 224))

    # Convert the reshaped array back to an image
    img_to_save = Image.fromarray(img_array_reshaped[0].astype(np.uint8))

    # Save the image
    img_to_save.save(output_image_path)

def extract_images_from_grid(grid, epoch, output_dir, counter, num_images_per_row=8, padding=2):
    
    # print("Extracting from grid.")
    # Get the total size of the grid
    _, grid_height, grid_width = grid.size()
    
    # Calculate the size of each individual image
    image_size = (grid_height - (num_images_per_row + 1) * padding) // num_images_per_row
    
    # List to store individual images
    images = []
    # counter = 1
    
    for i in range(num_images_per_row):
        for j in range(num_images_per_row):
            x_start = j * (image_size + padding) + padding
            y_start = i * (image_size + padding) + padding
            x_end = x_start + image_size
            y_end = y_start + image_size
            
            # Extract the individual image
            image = grid[:, y_start:y_end, x_start:x_end]
            images.append(image)
            
            # Save the individual image
            image_path = os.path.join(output_dir, f'image_{epoch}_{counter}.png')
            save_image_as_1_224_224(image, image_path)
            # torchvision.utils.save_image(image, image_path)
            counter += 1
    
    # return images

def select_random(data, percent):
    # Group elements by class
    grouped = defaultdict(list)
    for item in data:
        grouped[item[1]].append(item)

    # Select 10% of each class
    selected = []
    for class_items in grouped.values():
        ten_percent_length = int(len(class_items) * percent)
        selected.extend(random.sample(class_items, ten_percent_length))

    return selected

def get_last_batch(data_loader):
    last_batch = None
    for batch in data_loader:
        last_batch = batch
    return last_batch