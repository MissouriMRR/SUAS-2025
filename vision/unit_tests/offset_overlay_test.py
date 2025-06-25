from vision.mapping.overlaying import offset_overlay
from vision.common import constants
import cv2
import numpy as np


image_path = "vision/unit_tests/blue_square.png"
image: constants.Image = cv2.imread(image_path)
image += 1
camera_parameters: constants.CameraParameters = []


map: constants.Image = cv2.imread("vision/unit_tests/red_square.png")
map += 1
map_distance = np.zeros((map.shape[0], map.shape[1])) + 1
distance = np.zeros((image.shape[0], image.shape[1])) + 1

# image=np.dstack((image,distance))
# print(image.shape)
# map=np.stack((map,map_distance),axis=2)
new_image: constants.ImageInfo = {
    "image_path": image_path,
    "image": image,
    "image_shape": {"width": image.shape[1], "height": image.shape[0], "channels": image.shape[2]},
    "camera_parameters": camera_parameters,
    "distance": distance,
}
overlayed_image = offset_overlay(new_image, map, map_distance, (-1000, 1000), 0)
cv2.imwrite("overlayed.png", overlayed_image[0])
