from pydoc import plain
import unittest
import numpy as np
from main import extract_features, SAMPLING_TIME

class TestFeatureExtraction(unittest.TestCase):
    PLAIN_WINDOW = [[1.0] * 16] * 7  # Constant value (flat line)
    NORMAL_WINDOW = [
        [-2.6675938e+003, -2.7077491e+003, -2.7479043e+003, -2.7880596e+003, -2.8282148e+003, -2.8683701e+003, -2.9085253e+003, -2.9486806e+003, -2.9888358e+003, -3.0289910e+003, -3.1377484e+003, -3.3385643e+003, -3.5393802e+003, -3.7401961e+003, -3.9410120e+003, -4.1418279e+003],
        [4.9297612e-005,  4.8639761e-005,  4.7981909e-005,  4.7324058e-005,  4.6666206e-005,  4.6008355e-005,  4.5350503e-005,  4.4692652e-005,  4.4034800e-005,  4.3376949e-005,  4.2634755e-005,  4.1779379e-005,  4.0924003e-005,  4.0068627e-005,  3.9213252e-005,  3.8357876e-005],
        [-1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000, -1.0000000e+000],
        [0.0000000e+000,  0.0000000e+000,  0.0000000e+000,  0.0000000e+000,  3.0698104e+016,  7.2993812e+016,  9.2123192e+016,  8.2562325e+016,  7.6861185e+016,  6.1418398e+016,  4.6521811e+016,  3.9712788e+016,  1.8178069e+016,  0.0000000e+000,  0.0000000e+000,  0.0000000e+000],
        [-8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000, -8.7604830e+000],
        [1.0369252e+004,  1.1222247e+004,  1.4785431e+004,  1.3574248e+004,  3.1740108e+004,  4.7607341e+004,  5.7555218e+004,  7.0313385e+004,  7.1876954e+004,  7.0895166e+004,  5.9682656e+004,  4.2277674e+004,  3.6698041e+004,  2.1047949e+004,  4.3944682e+003,  3.3549300e+003],
        [1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000,  1.0000000e+000],
    ]
    DISRUPTIVE_WINDOW = [
        [-3.4444430e+005, -3.4500643e+005, -3.4556857e+005, -3.4613070e+005, -3.4669284e+005, -3.4725498e+005, -3.4781711e+005, -3.4837925e+005, -3.4894138e+005, -3.4968091e+005, -3.5052417e+005, -3.5136743e+005, -3.5221069e+005, -3.5305395e+005, -3.5389721e+005, -3.5474047e+005],
        [1.0639376e-004,  1.0718313e-004,  1.0797250e-004,  1.0876187e-004,  1.0955124e-004,  1.1034061e-004,  1.1112998e-004,  1.1191935e-004,  1.1270872e-004,  1.1233583e-004,  1.1128327e-004,  1.1023071e-004,  1.0917815e-004,  1.0812558e-004,  1.0707302e-004,  1.0602046e-004],
        [1.8800211e-001,  1.8886280e-001,  1.9045018e-001,  1.9203756e-001,  1.9362494e-001,  1.9521233e-001,  1.9679971e-001,  1.9838709e-001,  1.9997447e-001,  2.0156186e-001,  2.0314924e-001,  2.0472944e-001,  2.0607727e-001,  2.0742510e-001,  2.0877294e-001,  2.1012077e-001],
        [1.1715286e+019,  1.1686620e+019,  1.1675997e+019,  1.1671088e+019,  1.1658629e+019,  1.1650574e+019,  1.1648528e+019,  1.1654799e+019,  1.1676947e+019,  1.1710048e+019,  1.1734142e+019,  1.1754255e+019,  1.1755786e+019,  1.1741246e+019,  1.1721070e+019,  1.1703393e+019],
        [9.6259061e+003,  1.2356178e+004,  1.8174927e+004,  2.3993676e+004,  2.9812426e+004,  3.5631175e+004,  4.1449925e+004,  4.7268674e+004,  5.3087424e+004,  5.8906173e+004,  6.4724922e+004,  7.0353472e+004,  6.9832234e+004,  6.9310995e+004,  6.8789757e+004,  6.8268518e+004],
        [1.9331242e+005,  1.8214115e+005,  1.7148085e+005,  1.7314152e+005,  1.9290593e+005,  1.8410119e+005,  1.4944622e+005,  1.2366120e+005,  1.4005295e+005,  1.6276332e+005,  2.0644618e+005,  2.3550505e+005,  2.2550106e+005,  1.9306008e+005,  1.5468456e+005,  1.4209620e+005],
        [7.4709236e+005,  7.4738787e+005,  7.4694831e+005,  7.4650875e+005,  7.4606919e+005,  7.4562962e+005,  7.4519006e+005,  7.4475050e+005,  7.4431094e+005,  7.4387137e+005,  7.4343181e+005,  7.4298026e+005,  7.4214084e+005,  7.4130142e+005,  7.4046201e+005,  7.3962259e+005]
    ]

    def test_window_size(self):
        """Test that provided windows have the correct size."""
        plain_window = np.array(self.PLAIN_WINDOW)
        normal_window = np.array(self.NORMAL_WINDOW)
        disruptive_window = np.array(self.DISRUPTIVE_WINDOW)
        self.assertEqual(plain_window.shape, (7, 16), "Plain window should have shape (7, 16)")
        self.assertEqual(normal_window.shape, (7, 16), "Normal window should have shape (7, 16)")
        self.assertEqual(disruptive_window.shape, (7, 16), "Disruptive window should have shape (7, 16)")

    def test_extract_features_produces_different_outputs(self):
        """Test that different windows produce different feature vectors."""
        # Extract features for each window
        features_plain = extract_features(self.PLAIN_WINDOW)
        features_normal = extract_features(self.NORMAL_WINDOW)
        features_disruptive = extract_features(self.DISRUPTIVE_WINDOW)

        # Verify each feature vector has the expected shape
        self.assertEqual(features_plain.shape, (1, 7), "Feature vector should have 7 features")
        self.assertEqual(features_normal.shape, (1, 7), "Feature vector should have 7 features")
        self.assertEqual(features_disruptive.shape, (1, 7), "Feature vector should have 7 features")
        
        # Verify the plain window has specific characteristics
        self.assertAlmostEqual(features_plain[0][0], 1.0, msg="Mean of plain window should be 1.0")
        self.assertAlmostEqual(features_plain[0][1], 0.0, msg="Slope of plain window should be 0.0")
        
        # Ensure all three feature vectors are different from each other
        self.assertFalse(np.array_equal(features_plain, features_normal), 
                        "Plain and normal window features should be different")
        self.assertFalse(np.array_equal(features_plain, features_disruptive), 
                        "Plain and disruptive window features should be different")
        self.assertFalse(np.array_equal(features_normal, features_disruptive), 
                        "Normal and disruptive window features should be different")
    
    def test_extract_features_individual_components(self):
        """Test that each feature extraction component works as expected."""
        # Test on the disruptive window as it has interesting characteristics
        features = extract_features(self.DISRUPTIVE_WINDOW)
        
        # Manually calculate expected values
        window = np.array(self.DISRUPTIVE_WINDOW)
        expected_mean = np.mean(window)
        diff = np.diff(window)
        expected_slope = diff.mean() * 1/SAMPLING_TIME
        expected_log_rms = np.log1p(np.sqrt(np.mean(np.square(window))))
        expected_max_slope = np.max(np.abs(diff))
        abs_second_deriv = np.abs(np.diff(diff))
        expected_min_2nd_deriv = np.min(abs_second_deriv) * 1/(SAMPLING_TIME**2)
        expected_max_2nd_deriv = np.max(abs_second_deriv) * 1/(SAMPLING_TIME**2)
        expected_mean_2nd_deriv = np.mean(abs_second_deriv) * 1/(SAMPLING_TIME**2)
        
        # Compare with calculated features
        self.assertAlmostEqual(features[0][0], expected_mean, places=6, 
                               msg="Mean calculation is incorrect")
        self.assertAlmostEqual(features[0][1], expected_slope, places=6, 
                               msg="Slope calculation is incorrect")
        self.assertAlmostEqual(features[0][2], expected_log_rms, places=6, 
                               msg="Log-RMS calculation is incorrect")
        self.assertAlmostEqual(features[0][3], expected_max_slope, places=6, 
                               msg="Max slope calculation is incorrect")
        self.assertAlmostEqual(features[0][4], expected_min_2nd_deriv, places=6, 
                               msg="Min 2nd derivative calculation is incorrect")
        self.assertAlmostEqual(features[0][5], expected_max_2nd_deriv, places=6, 
                               msg="Max 2nd derivative calculation is incorrect")
        self.assertAlmostEqual(features[0][6], expected_mean_2nd_deriv, places=6, 
                               msg="Mean 2nd derivative calculation is incorrect")

if __name__ == "__main__":
    unittest.main()
