import torch.nn as nn
import cv2
import numpy as np

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