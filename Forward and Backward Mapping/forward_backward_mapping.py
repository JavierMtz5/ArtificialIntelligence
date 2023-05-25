from typing import Tuple
import time

import matplotlib.pyplot as plt
import numpy as np
from PIL import Image


def apply_transformation(original_x: int, original_y: int) -> Tuple[int, int]:
    # Define the rotation matrix
    rotate_transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],
                                      [np.sin(np.pi/4),  np.cos(np.pi/4), 0],
                                      [0, 0, 1]])
    # Apply transformation after setting homogenous coordinate to 1 for the original vector.
    new_coordinates = rotate_transformation @ np.array([original_x, original_y, 1]).T
    # Round the new coordinates to the nearest pixel
    return int(np.rint(new_coordinates[0])), int(np.rint(new_coordinates[1]))


def apply_inverse_transformation(new_x: int, new_y: int) -> Tuple[int, int]:
    # Define the inverse rotation matrix
    rotate_transformation = np.array([[np.cos(np.pi/4), -np.sin(np.pi/4), 0],
                                      [np.sin(np.pi/4),  np.cos(np.pi/4), 0],
                                      [0, 0, 1]])
    inverse_rotate_transformation = np.linalg.inv(rotate_transformation)
    # Apply transformation after setting homogenous coordinate to 1 for the position vector.
    original_coordinates = inverse_rotate_transformation @ np.array([new_x, new_y, 1]).T
    # Round the original coordinates to the nearest pixel
    return int(np.rint(original_coordinates[0])), int(np.rint(original_coordinates[1]))


def forward_mapping(original_image: np.ndarray) -> np.ndarray:
    # Create the new image with same shape as the original one
    new_image = np.zeros_like(original_image)
    for original_y in range(original_image.shape[1]):
        for original_x in range(original_image.shape[0]):
            # Apply rotation on the original pixel's coordinates
            new_x, new_y = apply_transformation(original_x, original_y)
            # Check if new coordinates fall inside the image's domain
            if 0 <= new_y < new_image.shape[1] and 0 <= new_x < new_image.shape[0]:
                new_image[new_x, new_y, :] = original_image[original_x, original_y, :]

    return new_image


def backward_mapping(original_image: np.ndarray) -> np.ndarray:
    # Create the new image with same shape as the original one
    new_image = np.zeros_like(original_image)
    for new_y in range(new_image.shape[1]):
        for new_x in range(new_image.shape[0]):
            # Apply inverse rotation on the new pixel's coordinates
            original_x, original_y = apply_inverse_transformation(new_x, new_y)
            # Check if original coordinates fall inside the image's domain
            if 0 <= original_y < original_image.shape[1] and 0 <= original_x < original_image.shape[0]:
                new_image[new_x, new_y, :] = original_image[original_x, original_y, :]

    return new_image


if __name__ == '__main__':

    # Load image
    original_image = Image.open('handwritten_digit.png')
    original_image = np.asarray(original_image)
    original_image = original_image / 255.

    # Apply rotation on the image with Forward Mapping
    start = time.time()
    new_image = forward_mapping(original_image)
    print(f'Forward Mapping was executed in {time.time() - start} seconds')
    plt.imshow(new_image)
    plt.show()

    # Apply rotation on the image with Backward Mapping
    start = time.time()
    new_image = backward_mapping(original_image)
    print(f'Backward Mapping was executed in {time.time() - start} seconds')
    plt.imshow(new_image)
    plt.show()
