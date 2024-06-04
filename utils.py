import os
import cv2
import numpy as np
import torch.nn as nn
import torchvision

def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        nn.init.normal_(m.weight.data, 0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        nn.init.normal_(m.weight.data, 1.0, 0.02)
        nn.init.constant_(m.bias.data, 0)

def resize_image(image, size):
    image = cv2.resize(image, (size, size), cv2.INTER_NEAREST)
    return image

def concat_image(images):
    b, h, w, c = images.shape
    num_side = int(np.sqrt(b))
    image = np.vstack([np.hstack(images[i * num_side:(i + 1) * num_side]) for i in range(num_side)])
    return image


def save_image(file_name, image):
    image = np.array((image + 1) * 127.5, dtype="uint8")
    cv2.imwrite(file_name, image)

def extract_images_from_grid(grid, epoch, output_dir, num_images_per_row=8, padding=2):
    
    # print("Extracting from grid.")
    # Get the total size of the grid
    _, grid_height, grid_width = grid.size()
    
    # Calculate the size of each individual image
    image_size = (grid_height - (num_images_per_row + 1) * padding) // num_images_per_row
    
    # List to store individual images
    images = []
    counter = 0
    
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
            torchvision.utils.save_image(image, image_path)
            counter += 1
    
    # return images