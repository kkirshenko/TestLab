import unittest
import sys
from app import Kernel, ImageConvolution


class TestImageConvolutionAutomated(unittest.TestCase):

    def test_valid_kernel_creation(self):
        # Автоматическая проверка создания ядра
        valid_matrix = [[0, 1, 0], [1, 1, 1], [0, 1, 0]]
        kernel = Kernel("blur", valid_matrix)
        self.assertEqual(kernel.name, "blur")
        self.assertEqual(len(kernel.matrix), 3)

    def test_kernel_non_square_matrix(self):
        # Автоматическая проверка перехвата ошибки для неквадратной матрицы.
        non_square_matrix = [[1, 0, -1], [1, 0, -1]]
        with self.assertRaises(ValueError) as context:
            Kernel("invalid", non_square_matrix)
        self.assertIn("Матрица ядра должна быть квадратной", str(context.exception))

    def test_kernel_even_size(self):
        # Автоматическая проверка перехвата ошибки для ядра четного размера.
        even_matrix = [[1, 0], [0, 1]]
        with self.assertRaises(ValueError) as context:
            Kernel("invalid_even", even_matrix)
        self.assertIn("Размер ядра должен быть нечетным", str(context.exception))

    def test_apply_identity_kernel(self):
        # Автоматическая проверка математической корректности свертки.
        input_image = [
            [10, 20, 30],
            [40, 50, 60],
            [70, 80, 90]
        ]

        valid_identity = [
            [0, 0, 0],
            [0, 1, 0],
            [0, 0, 0]
        ]
        kernel = Kernel("identity_3x3", valid_identity)

        convolution = ImageConvolution(kernel)
        result_image = convolution.apply(input_image)

        self.assertEqual(result_image, input_image)


if __name__ == '__main__':

    print("Запуск автоматизированного тестирования...\n")
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromTestCase(TestImageConvolutionAutomated))

    with open("test_protocol.txt", "w", encoding="utf-8") as f:
        f.write("=== ПРОТОКОЛ АВТОМАТИЗИРОВАННОГО ТЕСТИРОВАНИЯ ===\n")
        f.write(f"Приложение: Свертка изображения (Вариант 5)\n")
        f.write(f"Всего выполнено тестов: {result.testsRun}\n")
        f.write(f"Успешно пройдено: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Ошибок выполнения (Errors): {len(result.errors)}\n")
        f.write(f"Провалов проверок (Failures): {len(result.failures)}\n")
        f.write("-" * 50 + "\n")
        if result.wasSuccessful():
            f.write("ИТОГ: Все тесты пройдены успешно. Приложение работает корректно.\n")
        else:
            f.write("ИТОГ: Обнаружены дефекты. Требуется доработка кода.\n")

    print("\nПротокол тестирования сохранен в файл 'test_protocol.txt'")