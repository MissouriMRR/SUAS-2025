import numpy as np
import vision.common.constants as consts
from PIL import Image
import matplotlib.pyplot as plt

from vision.deskew import camera_distances
from vision.deskew import deskew
from vision.deskew import coordinate_lengths
from vision.deskew import vector_utils

from vision.mapping import overlaying

import cv2
import math
import logging


class Map:
    """
    An object representing a map
    Stores the map image itself, the scale of the map, and the bounds on the coordinates of the map

    Parameters
    ----------
    pixels_per_foot: float
        The scale of the map in pixel *edges* per foot - for example, a value of 2 means that one square foot will
        take up a square of 4 pixels
    feather_width: float
        Roughly the width of the fade between adjacent images
    img: consts.MapImage
        The map image. Alpha channel will be used to store resolution data
    coord_min: consts.Point
        The minimum lat, lon of the map
    coord_max: consts.Point
        The maximum lat, lon of the map
    """

    def __init__(self, pixels_per_foot, feather_width=10) -> None:
        self.pixels_per_foot: float = pixels_per_foot
        self.feather_width = feather_width

        self.map_distance = None
        self.img: consts.MapImage = None
        self.img_count = 0
        self.coord_min = None
        self.coord_max = None

    def prepare_image(self, img: consts.ImageInfo):

        blank_channel = np.zeros((img["image_shape"]["height"], img["image_shape"]["width"]))
        blank_channel = blank_channel + 1
        img["distance"] = blank_channel
        # print("yes", self.pixels_per_foot, consts.FEET_PER_METER,img["camera_parameters"]["altitude"])

        img["image"], img_corner_pixels = deskew.deskew(
            img, scale=self.pixels_per_foot * consts.FEET_PER_METER
        )

        img["image_shape"]["height"] = img["image"].shape[0]
        img["image_shape"]["width"] = img["image"].shape[1]
        self.fill_distances(img)

        return img["image"]

    def fill_distances(self, img: consts.ImageInfo) -> None:

        distance = lambda x, y: np.linalg.norm(
            vector_utils.pixel_intersect(
                (x, y),
                img["image"].shape,
                img["camera_parameters"]["rotation_deg"],
                height=img["camera_parameters"]["altitude"],
            )
        )
        height: int = img["image"].shape[0] - 1
        width: int = img["image"].shape[1] - 1
        center_height: int = img["image"].shape[0] // 2
        center_width: int = img["image"].shape[1] // 2
        # has the pixel coordinates of the start and end indexes of each fourth of the image

        # calculating distance for each pixel takes way too long, so it will be simplified by making the points up to the center on top,bottom, and center linearly scaled
        # then filling the rest linearly

        # initialize corners and center distances
        img["distance"] = np.zeros((img["image_shape"]["height"], img["image_shape"]["width"]))
        img["distance"][0, 0] = distance(0, 0)
        img["distance"][0, width] = distance(width, 0)
        img["distance"][height, width] = distance(width, height)
        img["distance"][height, 0] = distance(0, height)

        img["distance"][center_height, center_width] = distance(center_width, center_height)
        img["distance"][0, center_width] = distance(center_width, 0)
        img["distance"][center_height, 0] = distance(0, center_height)
        img["distance"][height, center_width] = distance(center_width, height)
        img["distance"][center_height, width] = distance(width, center_height)

        # start by filling top, bottom, and center
        for i in [0, center_height, height]:
            for pixel in range(1, width):
                if pixel < center_width:
                    img["distance"][i, pixel] = int(
                        img["distance"][i, 0] * pixel / (center_width)
                        + img["distance"][i, center_width] * (1 - pixel / (center_width))
                    )
                if pixel > center_width:
                    # percentages are close enough not perfect
                    img["distance"][i, pixel] = int(
                        img["distance"][i, center_width] * (pixel - center_width) / (center_width)
                        + img["distance"][i, width] * (1 - (pixel - center_width) / (center_width))
                    )

        # fill the rest
        for column in range(0, img["image"].shape[1]):
            for pixel in range(1, height):
                if pixel < center_height:
                    img["distance"][pixel, column] = int(
                        img["distance"][0, column] * pixel / (center_height)
                        + img["distance"][center_height, column] * (1 - pixel / (center_height))
                    )
                if pixel > center_height:
                    # percentages are close enough not perfect
                    img["distance"][pixel, column] = int(
                        img["distance"][center_height, column]
                        * (pixel - center_height)
                        / (center_height)
                        + img["distance"][height, column]
                        * (1 - (pixel - center_height) / (center_height))
                    )
        return

    def add_img(self, img: consts.ImageInfo):

        img["corner_coords"] = camera_distances.corner_coords(
            img["image"].shape, img["camera_parameters"]
        )
        logging.debug("corner coords: {}".format(img["corner_coords"]))
        projected_image_wdepth = self.prepare_image(img)

        if self.img_count == 0:

            img_min_coord = np.min(img["corner_coords"], axis=0)
            img_max_coord = np.max(img["corner_coords"], axis=0)
            self.img = projected_image_wdepth
            self.coord_min = img_min_coord
            self.coord_max = img_max_coord
            self.map_distance = img["distance"]
        else:
            self.add_projected_image(img)

        cv2.imwrite(f"zmap/map{self.img_count}.png", self.img)
        self.img_count += 1

        return

    def add_projected_image(self, proj_img: consts.ImageInfo):
        # Calculate the "top left" and "bottom right" corners of the bounding box in GPS coordinate space
        # May not actually be top left and bottom right, but it doesn't really matter - we just care about the bounds

        img_min_coord = np.min(proj_img["corner_coords"], axis=0)
        img_max_coord = np.max(proj_img["corner_coords"], axis=0)

        # Calculate the center of the projected image
        center_coord = (img_min_coord + img_max_coord) / 2

        map_center = (self.coord_max + self.coord_min) / 2

        # The position of the projected image relative to the map in GPS coordinate space
        coordinate_change = center_coord - map_center

        # Calculate the latitude and longitude lengths (in meters)
        latitude_length: float = coordinate_lengths.latitude_length(center_coord[0])
        longitude_length: float = coordinate_lengths.longitude_length(center_coord[0])

        relative_coord_ft = np.array(
            [coordinate_change[0] * longitude_length, coordinate_change[1] * latitude_length]
        )

        print("PIXELS PER FOOT", self.pixels_per_foot)
        pixel_offset = np.round(relative_coord_ft * self.pixels_per_foot)
        pixel_offset[1] *= -1  # negative is up
        updated_map_img, self.map_distance = overlaying.offset_overlay(
            proj_img, self.img, self.map_distance, pixel_offset, self.feather_width
        )
        """
        input("1")
        print("img min coord",img_min_coord)
        print("img max coord",img_max_coord)
        print("map min coord",self.coord_min)
        print("map max coord",self.coord_max)

        print("center coord ",center_coord)
        print("map center",map_center)
        print("coordinate change",coordinate_change)
        print("latitude_length",latitude_length)
        print("long_length",longitude_length)
        print("pixel offset",pixel_offset)
        input("2")
        """

        self.coord_min = np.minimum(self.coord_min, img_min_coord)
        self.coord_max = np.maximum(self.coord_max, img_max_coord)
        print("min coord", self.coord_min)
        self.img = updated_map_img

        return
