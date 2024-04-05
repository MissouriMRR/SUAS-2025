import unittest
import numpy as np
import json
import chars
from odlc_classify_shape import cart2pol, toPolar, merge_sort, condense_polar, Generate_Polar_Array, Verify_Shape_Choice, Compare_Based_On_Peaks, classify_shape, Image_Address_To_Contour, classify_shap

class TestShapeClassification(unittest.TestCase):
'''
    # Convertes x and y rectangular values to radius, angle tuples
    def test_cart2pol(self):
        x, y = 3, 4
        expected_rho = 5.0
        expected_phi = np.arctan(4, 3)
        rho, phi = cart2pol(x, y)
        self.assertAlmostEqual(rho, expected_rho)
        self.assertAlmostEqual(phi, expected_phi)


    
    # Test the toPolar function that converts an array of rectangular coordinates to polar coordinates    def test_toPolar(self):
        input_data = [((0, 0),), ((1, 0),), ((0, 1),), ((-1, 0),), ((0, -1),)]
        expected_output = [(0.0, 0.0), (1.0, 0.0), (1.0, np.pi/2), (1.0, np.pi), (1.0, -np.pi)]
        for i in range(len()):
            {
                for i, coords in enumerate(input_data):
                    with self.subTest (i =i):
                        result = toPolar(np.array(coords))
                        self.assertAlmostEqual(result[0][0], expected_output[i][0], places=8)
                        self.assertAlmostEqual(result[0][1], expected_output[i][1], places=8)
            }
# Python implementation of merge sort algorithm, slightly edited to fit our array structure of tuples (sorted based on increasing angle).
    def test_merge_sort(self):
        # Test cases
        assert merge_sort([]) == []
        assert merge_sort([(1, 5), (2, 3), (3, 1)]) == [(3, 1), (2, 3), (1, 5)]
        assert merge_sort([(4, 2), (1, 7), (3, 4), (2, 6)]) == [(3, 4), (4, 2), (2, 6), (1, 7)]

# Condenses the array of polar coordinates to have 'NUM_STEPS' points stored for analysis    def Generate_Polar_Array(self):
    def test_condense_polar(self):
        polar_array = [(1.0, 0.0), (1.0, np.pi/2), (1.0, np.pi), (1.0, -np.pi)]
        newx, newy = condense_polar(polar_array)
        expected_newx = np.linspace(-np.pi, np.pi, num=128)
        expected_newy = np.array([1.0, 1.0, 1.0, 1.0])  # Adjust this based on your expected output
        np.testing,assert_allclose(newx, expected_newx, rtol=1e-8)
        np.testing.assert_allclose(newy, expected_newy, rtol=1e-8)

# Checks to see if an ODLC's Polar graph is adequately similar to the "guessed" ODLC's sample graph
    def Verify_Shape_Choice(self):
        # Test case
        mystery_radii_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        sample_ODLC_radii = [0.11, 0.19, 0.31, 0.39, 0.51, 0.59, 0.71, 0.79]
        shape_choice = 3  # Assuming triangle
        assert Verify_Shape_Choice(mystery_radii_list, shape_choice, sample_ODLC_radii) is True
        
        # Test case 2: Triangle shape ( pass)
        mystery_radii_list = [0.1, 0.2, 0.3, 0.4 0.5,0.6,0.7,0.8]:
        sample_ODLC_radii = [0.12, 0.18, 0.32, 0.38, 0.52, 0.58, 0.72, 0.78]

        shape_choice = 2 # Assuming Circle
        assert Verify_Shape_Choice(mystery_radii_list, shape_choice, sample_ODLC_radii) is False

        # Test case 3: 


    def Compare_Based_On_Peaks(self):
           
        mock_find_peaks.return_value = ([1, 3, 5],)
        
        # Mock the return value of json.load
        mock_load.return_value = {
            "CIRCLE": [(1.0, 0.0)],
            "TRIANGLE": [(1.0, 0.0), (1.0, 2.0943951023931957), (1.0, -2.0943951023931957)],
'''
    
    def test_classify_shape(self):
    
        classifier = YourShapeClassifierClass()

        for i in range(8):
            contour_sample = "/Users/ouyangyuxuan/Desktop/NewImg/shape" + str(i + 1) + ".png"
            contour_sample_image = classifier.countour_address(contour_sample) 
            image_dims = (contour_sample_image.shape[0], contour_sample_image.shape[1])  
            shape = classifier.classify_shape(contour_sample_image, image_dims)

        def test_shape_mapping(self):
            expected_mapping = {
                ODLCShape.CIRCLE: 0,
                ODLCShape.QUARTER_CIRCLE: 1,
                ODLCShape.SEMICIRCLE: 2,
                ODLCShape.TRIANGLE: 3,
                ODLCShape.PENTAGON: 4,
                ODLCShape.STAR: 5,
                ODLCShape.RECTANGLE: 6,
                ODLCShape.CROSS: 7
            }

            self.assertEqual(ODLCShape_To_ODLC_Index, expected_mapping)
``
                for shape, path in image_paths.items():
            # Load the contour from the image path (you may need to adjust this based on your implementation)
            contour = Image_Address_To_Contour(path)
            # Assume image dimensions for this example
            image_dims = (200, 200)
            
            # Call the classify_shape function with the contour
            result = classify_shape(contour, image_dims)
            
            # Assert that the result is the expected shape based on the key in image_paths
            expected_shape = ODLCShape[shape.upper()]  # Assuming shape names are all uppercase
            self.assertEqual(result, expected_shape)












'''

    def test_verify_shape_choice(self):

        mystery_radii_list = [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]
        shape_choice = chars.ODLCShape.CIRCLE

        sample_ODLC_radii = [0.11, 0.22, 0.33, 0.44, 0.55, 0.66, 0.77, 0.88]
        self.assertTrue(Verify_Shape_Choice(mystery_radii_list, shape_choice, sample_ODLC_radii))


''' 
        
    if __name__ == '__main__':
        unittest.main()