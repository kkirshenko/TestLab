import unittest
from app import Kernel, ImageConvolution


class TestFunctionalRequirements(unittest.TestCase):

    def test_kernel_creation_valid(self):
        # Требование 1. Ядро должно успешно создаваться, если оно квадратное и нечетное.
        kernel = Kernel("blur", [[0, 1, 0], [1, 1, 1], [0, 1, 0]])
        self.assertEqual(kernel.name, "blur")
        self.assertEqual(len(kernel.matrix), 3)

    def test_kernel_creation_invalid_even(self):
        # Требование 2. Ядро четного размера должно вызывать ошибку валидации.
        with self.assertRaises(ValueError):
            Kernel("invalid", [[1, 0], [0, 1]])

    def test_convolution_clipping_max(self):

        amplify_kernel = Kernel("amplify", [[10]])
        convolution = ImageConvolution(amplify_kernel)

        input_image = [[30]]
        result = convolution.apply(input_image)

        self.assertEqual(result[0][0], 255)

    def test_convolution_clipping_min(self):

        negative_kernel = Kernel("negative", [[-1]])
        convolution = ImageConvolution(negative_kernel)

        input_image = [[50]]
        result = convolution.apply(input_image)

        self.assertEqual(result[0][0], 0)


if __name__ == '__main__':
    unittest.main()