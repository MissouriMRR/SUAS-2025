import numpy as np
import cv2
from vision.common.constants import ImageInfo, Image

# Coordinate system:
#  __ +y axis
# |
# +x axis


def calculate_padding(img1, img2, r12):
    # Pre-defined arbitrary offset r12: img1 top left corner - img2 top left corner
    """
    Calculates padding for adding img1 to img2
    r12 = (offset_x, offset_y) = img1_topleft - img2_topleft
    """

    # Only need to pad img2

    # img2 top left padding:
    # Only pad top left if some axis of r12 is negative
    img2_top_padding = max(0, (img1.shape[0] - img2.shape[0]) / 2 - r12[1])
    img2_left_padding = max(0, (img1.shape[1] - img2.shape[1]) / 2 - r12[0])

    # Only pad bottom right if overlayed img1 will go beyond the bottom/right of img2
    img2_bottom_padding = max(0, (img1.shape[0] - img2.shape[0]) / 2 + r12[1])
    img2_right_padding = max(0, (img1.shape[1] - img2.shape[1]) / 2 + r12[0])

    return img2_top_padding, img2_bottom_padding, img2_left_padding, img2_right_padding


def pad_image(img, padding, value):
    padded = cv2.copyMakeBorder(
        img,
        int(padding[0] + 0.5),
        int(padding[1] + 0.5),
        int(padding[2] + 0.5),
        int(padding[3] + 0.5),
        cv2.BORDER_CONSTANT,
        value=value,
    )

    return padded


def alpha_over(img1, img2, alpha_channel):
    """
    Overlays img1 on top of img2 using alpha_channel
    """

    # overlayed=img1* alpha_channel[:,:,None]
    # overlayed =
    overlayed = img1 * alpha_channel[:, :, None] + (1 - alpha_channel[:, :, None]) * img2

    return img1


def rgb2rgba(img):
    img_alpha = np.ones((img.shape[0], img.shape[1])) * 255

    img_w_alpha = np.dstack((img, img_alpha))

    return img_w_alpha


def distance_alpha(distance1, distance2, feather_width):
    # Sigmoid function
    return (np.tanh(4 * (distance1 - distance2) / feather_width) + 1) / 2


def layering(new_img: ImageInfo, padded_section, padded_distance_section):
    # this works but doesn't look the best
    """
    for x in range(new_img["image"].shape[1]):
        for y in range(new_img["image"].shape[0]):
            print(new_img["image"][y,x,:])
            if new_img["image"][y,x,2] is None or new_img["image"][y,x,2] is 0:
                new_img["image"][y,x,:] = padded_section[y,x,:]
                new_img["distance"][y,x] = padded_distance_section[y,x]
            elif padded_section[y,x,2] == None or padded_section[y,x,2] == 0:
                #have new_img[x,y,3] = new_img[x,y,3]
                continue
            else:
                # needs to be changed to use distances but whole map image will also need to be changed
                if(new_img["distance"][y,x] > padded_distance_section[y,x] or new_img["distance"][y,x] == 0 or new_img["distance"][y,x] == None):
                    new_img["distance"][y,x] = padded_distance_section[y,x]
                    new_img["image"][y,x,:] = padded_section[y,x,:]
    """
    # for some reason this just works better

    for x in range(new_img["image"].shape[1]):
        for y in range(new_img["image"].shape[0]):
            if new_img["image"][y, x, 2] == None or new_img["image"][y, x, 2] == 0:

                new_img["image"][y, x, :] = padded_section[y, x, :]
                new_img["distance"][y, x] = padded_distance_section[y, x]
            elif padded_section[y, x, 2] == None or padded_section[y, x, 2] == 0:
                # have new_img[x,y,3] = new_img[x,y,3]
                continue
            else:
                new_img["image"][y, x, :] = (
                    padded_section[y, x, :] * 0.5 + new_img["image"][y, x, :] * 0.5
                )

    return new_img["image"], new_img["distance"]


def offset_overlay(new_img: ImageInfo, base_img: Image, map_distance, new_location, feather_width):

    padding = calculate_padding(new_img["image"], base_img, new_location)

    padded = pad_image(base_img, padding, np.array([0, 0, 0, np.inf]))  # Zero for color channels,
    padded_distance = pad_image(map_distance, padding, 500)

    # img2_top_padding, img2_bottom_padding, img2_left_padding, img2_right_padding
    # New offset relative to the padded image
    shifted_new_location = [
        new_location[0] - (padding[3] - padding[2]) / 2,
        new_location[1] - (padding[1] - padding[0]) / 2,
    ]

    top_left = [
        int((padded.shape[0] / 2 + shifted_new_location[1]) - new_img["image"].shape[0] / 2),
        int((padded.shape[1] / 2 + shifted_new_location[0]) - new_img["image"].shape[1] / 2),
    ]

    # calculate the bottom left corner of the region to overlay on
    bottom_right = top_left + np.array(new_img["image"].shape[:2])

    # Get the section of the padded image where img1 will be overlayed
    padded_section = padded[
        int(top_left[0]) : int(bottom_right[0]), int(top_left[1]) : int(bottom_right[1]), :
    ]
    padded_distance_section = padded_distance[
        int(top_left[0]) : int(bottom_right[0]), int(top_left[1]) : int(bottom_right[1])
    ]

    # alpha_channel = distance_alpha(new_img[:, :, 3], padded_section[:, :, 3], feather_width)

    # overlayed_section = alpha_over(new_img, padded_section, alpha_channel)
    cv2.imwrite("paddedtest.png", padded_section)
    cv2.imwrite("new_imgtest.png", new_img["image"])
    overlayed_section, overlayed_distance_section = layering(
        new_img, padded_section, padded_distance_section
    )

    # Write the overlayed section back to the image
    padded[int(top_left[0]) : int(bottom_right[0]), int(top_left[1]) : int(bottom_right[1]), :] = (
        overlayed_section
    )
    padded_distance[
        int(top_left[0]) : int(bottom_right[0]), int(top_left[1]) : int(bottom_right[1])
    ] = overlayed_distance_section

    return padded, padded_distance


def main():
    img1 = cv2.imread("test.png", cv2.IMREAD_UNCHANGED)

    img2 = cv2.imread("goatthing.jpg")
    img2 = rgb2rgba(img2)

    offset = np.array([-200, 50])

    padded = offset_overlay(img1, img2, offset)

    cv2.imwrite("padded.png", padded)


if __name__ == "__main__":
    main()
