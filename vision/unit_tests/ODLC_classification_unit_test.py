import unittest
import numpy as np
import cv2
from vision.common import constants as consts
from vision.common import odlc_characteristics as chars
from vision.standard_object.odlc_classify_shape import cart2pol, toPolar, merge_sort, condense_polar, Generate_Polar_Array, Verify_Shape_Choice, Compare_Based_On_Peaks, classify_shape

class TestShapeClassification(unittest.TestCase):

    def test_classify_shape(self):
        expected_mapping = {
            0: chars.ODLCShape.CIRCLE,
            1: chars.ODLCShape.QUARTER_CIRCLE,
            2: chars.ODLCShape.SEMICIRCLE,
            3: chars.ODLCShape.TRIANGLE,
            4: chars.ODLCShape.PENTAGON,
            5: chars.ODLCShape.STAR,
            6: chars.ODLCShape.RECTANGLE,
            7: chars.ODLCShape.CROSS
        }
        for i in range(8):
            file_name = "vision/unit_tests/test_images/standard_objects/shape" + str(i + 1) + ".png"
            image = cv2.imread(file_name, cv2.IMREAD_GRAYSCALE)
        
            ret, image = cv2.threshold(image, 190, 255, cv2.THRESH_BINARY)
            #image = cv2.blur(image, (5,5)) # Currently not using, but could be useful in the future for reducing noise
            
            contours, _hierarchy = cv2.findContours(image, cv2.RETR_TREE, cv2.CHAIN_APPROX_NONE) #, 2, 1

            cnt = contours[0] # Sets cnt equal to most outer Contours
            cv2.drawContours(image, cnt, -1, (100, 100, 100), 3)      
            contour_sample = cnt
            image_dims = contour_sample.shape
            shape = classify_shape(contour_sample, image_dims)

            self.assertEqual(shape, expected_mapping[i])
       
       
    if __name__ == '__main__':
        unittest.main()