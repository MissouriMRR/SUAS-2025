
import unittest
import numpy as np
from vision.common import constants as consts
from vision.common import odlc_characteristics as chars
from odlc_classify_shape import cart2pol, toPolar, merge_sort, condense_polar, Generate_Polar_Array, Verify_Shape_Choice, Compare_Based_On_Peaks, classify_shape, Image_Address_To_Contour, classify_shape


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
          contour_sample = Image_Address_To_Contour("vision/unit_tests/test_images/shape" + str(i + 1) + ".png")
          image_dims = contour_sample
          
          shape = classify_shape(contour_sample, image_dims)


          self.assertEqual(shape, expected_mapping[i])
      
          
  if __name__ == '__main__':
      unittest.main()

      