import unittest
from app import Kernel, ImageConvolution


class TestKernelWhiteBox(unittest.TestCase):

    def test_valid_kernel_creation(self):
        valid_matrix = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        kernel = Kernel("blur", valid_matrix)

        self.assertEqual(kernel.name, "blur")
        self.assertEqual(len(kernel.matrix), 3)

    def test_kernel_non_square_matrix(self):
        non_square_matrix = [[1, 0, -1], [1, 0, -1]]

        with self.assertRaises(ValueError) as context:
            Kernel("invalid", non_square_matrix)


        self.assertIn("Матрица ядра должна быть квадратной", str(context.exception))

    def test_kernel_even_size(self):

        even_matrix = [[1, 0], [0, 1]]

        with self.assertRaises(ValueError) as context:
            Kernel("invalid_even", even_matrix)

        self.assertIn("Размер ядра должен быть нечетным", str(context.exception))


class TestImageConvolutionWhiteBox(unittest.TestCase):

    def test_apply_identity_kernel(self):

        input_image = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ]

        identity_kernel = Kernel("identity", [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        convolution = ImageConvolution(identity_kernel)
        result_image = convolution.apply(input_image)

        expected_image = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ]

        self.assertEqual(result_image, expected_image)

if __name__ == '__main__':
    unittest.main()