import unittest
import sys
from app import Kernel, ImageConvolution


class TestKernelUnit(unittest.TestCase):
    """Unit-тесты для класса Kernel"""

    def test_create_valid_3x3_kernel(self):
        """Тест создания валидного ядра 3x3"""
        matrix = [[1, 0, -1], [1, 0, -1], [1, 0, -1]]
        kernel = Kernel("edge_detect", matrix)

        self.assertEqual(kernel.name, "edge_detect")
        self.assertEqual(len(kernel.matrix), 3)
        self.assertEqual(kernel.matrix, matrix)

    def test_create_valid_1x1_kernel(self):
        """Тест создания минимального ядра 1x1 (граничный случай)"""
        matrix = [[1]]
        kernel = Kernel("identity_1x1", matrix)

        self.assertEqual(kernel.name, "identity_1x1")
        self.assertEqual(len(kernel.matrix), 1)

    def test_create_valid_5x5_kernel(self):
        """Тест создания большего ядра 5x5"""
        matrix = [[1] * 5 for _ in range(5)]
        kernel = Kernel("blur_5x5", matrix)

        self.assertEqual(kernel.name, "blur_5x5")
        self.assertEqual(len(kernel.matrix), 5)

    def test_non_square_matrix_raises_error(self):
        """Тест обработки неквадратной матрицы"""
        non_square = [[1, 0, -1], [1, 0, -1]]

        with self.assertRaises(ValueError) as context:
            Kernel("invalid", non_square)

        self.assertIn("Матрица ядра должна быть квадратной", str(context.exception))

    def test_even_size_matrix_raises_error(self):
        """Тест обработки матрицы чётного размера"""
        even_matrix = [[1, 0], [0, 1]]

        with self.assertRaises(ValueError) as context:
            Kernel("invalid", even_matrix)

        self.assertIn("Размер ядра должен быть нечетным", str(context.exception))

    def test_empty_matrix_raises_error(self):
        """Тест обработки пустой матрицы (граничный случай)"""
        empty_matrix = []

        with self.assertRaises((ValueError, IndexError)):
            Kernel("invalid", empty_matrix)


class TestImageConvolutionUnit(unittest.TestCase):
    """Unit-тесты для класса ImageConvolution"""

    def test_identity_kernel_preserves_image(self):
        """Тест: ядро идентичности не изменяет изображение"""
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
        result = convolution.apply(input_image)

        self.assertEqual(result, input_image)

    def test_blur_kernel_averages_pixels(self):
        """Тест: нормализованное ядро размытия сохраняет яркость изображения"""
        input_image = [
            [100, 100, 100],
            [100, 100, 100],
            [100, 100, 100]
        ]
        blur_kernel = Kernel("blur", [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])
        convolution = ImageConvolution(blur_kernel)
        result = convolution.apply(input_image)

        for row in result:
            for pixel in row:
                self.assertEqual(pixel, 100)

    def test_clipping_max_value(self):
        """Тест: значения выше 255 обрезаются до 255"""
        input_image = [[250]]
        amplify_kernel = Kernel("amplify", [[2]])

        convolution = ImageConvolution(amplify_kernel)
        result = convolution.apply(input_image)

        self.assertEqual(result[0][0], 255)

    def test_clipping_min_value(self):
        """Тест: отрицательные значения обрезаются до 0"""
        input_image = [[50]]
        negative_kernel = Kernel("negative", [[-1]])

        convolution = ImageConvolution(negative_kernel)
        result = convolution.apply(input_image)

        self.assertEqual(result[0][0], 0)

    def test_single_pixel_image(self):
        """Тест: обработка изображения 1x1 (граничный случай)"""
        input_image = [[128]]
        identity_kernel = Kernel("identity", [[1]])

        convolution = ImageConvolution(identity_kernel)
        result = convolution.apply(input_image)

        self.assertEqual(result, [[128]])

    def test_large_uniform_image(self):
        """Тест: обработка большого однородного изображения"""
        size = 10
        input_image = [[200 for _ in range(size)] for _ in range(size)]

        blur_kernel = Kernel("blur", [
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9],
            [1 / 9, 1 / 9, 1 / 9]
        ])
        convolution = ImageConvolution(blur_kernel)
        result = convolution.apply(input_image)

        for row in result:
            for pixel in row:
                self.assertEqual(pixel, 200)

    def test_edge_detection_on_uniform_image(self):
        """Тест: обнаружение границ на однородном изображении"""
        input_image = [
            [100, 100, 100],
            [100, 100, 100],
            [100, 100, 100]
        ]
        edge_kernel = Kernel("edge", [
            [-1, -1, -1],
            [-1, 8, -1],
            [-1, -1, -1]
        ])

        convolution = ImageConvolution(edge_kernel)
        result = convolution.apply(input_image)

        for row in result:
            for pixel in row:
                self.assertEqual(pixel, 0)

    def test_zero_image_with_any_kernel(self):
        """Тест: нулевое изображение с любым ядром даёт нули"""
        input_image = [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ]
        blur_kernel = Kernel("blur", [
            [1, 1, 1],
            [1, 1, 1],
            [1, 1, 1]
        ])

        convolution = ImageConvolution(blur_kernel)
        result = convolution.apply(input_image)

        for row in result:
            for pixel in row:
                self.assertEqual(pixel, 0)


class TestConvolutionEdgeCases(unittest.TestCase):
    """Тесты граничных случаев для свёртки"""

    def test_2x2_image_with_3x3_kernel(self):
        """Тест: маленькое изображение (2x2) с большим ядром (3x3)"""
        input_image = [
            [50, 100],
            [150, 200]
        ]
        identity_kernel = Kernel("identity", [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ])

        convolution = ImageConvolution(identity_kernel)
        result = convolution.apply(input_image)

        self.assertEqual(len(result), 2)
        self.assertEqual(len(result[0]), 2)

    def test_all_zeros_kernel(self):
        """Тест: ядро из всех нулей"""
        input_image = [
            [100, 200],
            [150, 250]
        ]
        zero_kernel = Kernel("zero", [
            [0, 0, 0],
            [0, 0, 0],
            [0, 0, 0]
        ])

        convolution = ImageConvolution(zero_kernel)
        result = convolution.apply(input_image)

        for row in result:
            for pixel in row:
                self.assertEqual(pixel, 0)


if __name__ == '__main__':

    print("Запуск Unit-тестирования приложения «Свертка изображения»...\n")
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))

    with open("unit_test_protocol.txt", "w", encoding="utf-8") as f:
        f.write("=== ПРОТОКОЛ UNIT-ТЕСТИРОВАНИЯ ===\n")
        f.write("Приложение: Свертка изображения\n")
        f.write(f"Всего выполнено тестов: {result.testsRun}\n")
        f.write(f"Успешно пройдено: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Ошибок выполнения (Errors): {len(result.errors)}\n")
        f.write(f"Провалов проверок (Failures): {len(result.failures)}\n")
        f.write("-" * 50 + "\n")
        if result.wasSuccessful():
            f.write("ИТОГ: Все Unit-тесты пройдены успешно.\n")
        else:
            f.write("ИТОГ: Обнаружены дефекты.\n")

    print("\nПротокол сохранён в файл 'unit_test_protocol.txt'")