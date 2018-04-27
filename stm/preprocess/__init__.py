import numpy as np


def normalize(image):
    return (image-image.min())/(image.max()-image.min())