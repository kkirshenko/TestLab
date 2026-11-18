import unittest
import os
import tempfile
import subprocess
import sys
from app1 import Kernel, ImageConvolution, load_image_from_file, save_image_to_file, get_kernel_by_name


class TestDesktopFileOperations(unittest.TestCase):
    """Тестирование работы с файловой системой"""

    def setUp(self):
        """Создаём временный файл перед каждым тестом"""
        self.temp_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        self.temp_file.close()

    def tearDown(self):
        """Удаляем временный файл после каждого теста"""
        if os.path.exists(self.temp_file.name):
            os.remove(self.temp_file.name)

    def test_save_and_load_image(self):
        """Тест: сохранение и загрузка изображения"""
        image = [[10, 20, 30], [40, 50, 60], [70, 80, 90]]

        save_image_to_file(image, self.temp_file.name)

        loaded_image = load_image_from_file(self.temp_file.name)


        self.assertEqual(image, loaded_image)

    def test_load_nonexistent_file(self):
        """Тест: загрузка несуществующего файла"""
        with self.assertRaises(FileNotFoundError):
            load_image_from_file("nonexistent_file.csv")

    def test_load_empty_file(self):
        """Тест: загрузка пустого файла"""

        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("")

        with self.assertRaises(ValueError):
            load_image_from_file(self.temp_file.name)

    def test_load_invalid_data(self):
        """Тест: загрузка файла с некорректными данными"""

        with open(self.temp_file.name, 'w', encoding='utf-8') as f:
            f.write("abc,def,ghi\n")

        with self.assertRaises(ValueError):
            load_image_from_file(self.temp_file.name)


class TestDesktopCLI(unittest.TestCase):
    """Тестирование интерфейса командной строки"""

    def setUp(self):
        """Создаём временные файлы для тестов CLI"""
        self.input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        self.output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        self.input_file.close()
        self.output_file.close()

        # Запись тестового изображения во входной файл
        image = [[100, 100, 100], [100, 100, 100], [100, 100, 100]]
        save_image_to_file(image, self.input_file.name)

    def tearDown(self):
        """Удаление временные файлы"""
        for filepath in [self.input_file.name, self.output_file.name]:
            if os.path.exists(filepath):
                os.remove(filepath)

    def test_cli_blur_kernel(self):
        """Тест: запуск приложения с ядром размытия"""
        result = subprocess.run(
            [sys.executable, 'app1.py', self.input_file.name, self.output_file.name, '--kernel', 'blur'],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)
        self.assertIn("Обработка завершена", result.stdout)

        # Проверяем, что выходной файл создан
        self.assertTrue(os.path.exists(self.output_file.name))

    def test_cli_identity_kernel(self):
        """Тест: запуск приложения с ядром идентичности"""
        result = subprocess.run(
            [sys.executable, 'app1.py', self.input_file.name, self.output_file.name, '--kernel', 'identity'],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 0)

        result_image = load_image_from_file(self.output_file.name)
        expected_image = [[100, 100, 100], [100, 100, 100], [100, 100, 100]]
        self.assertEqual(result_image, expected_image)

    def test_cli_invalid_kernel(self):
        """Тест: запуск приложения с несуществующим ядром"""
        result = subprocess.run(
            [sys.executable, 'app1.py', self.input_file.name, self.output_file.name, '--kernel', 'invalid'],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Ошибка", result.stdout)

    def test_cli_missing_input_file(self):
        """Тест: запуск приложения с несуществующим входным файлом"""
        result = subprocess.run(
            [sys.executable, 'app1.py', 'nonexistent.csv', self.output_file.name, '--kernel', 'blur'],
            capture_output=True,
            text=True
        )

        self.assertEqual(result.returncode, 1)
        self.assertIn("Ошибка", result.stdout)


class TestDesktopKernelSelection(unittest.TestCase):
    """Тестирование выбора ядер"""

    def test_get_identity_kernel(self):
        """Тест: получение ядра идентичности"""
        kernel = get_kernel_by_name('identity')
        self.assertEqual(kernel.name, 'identity')
        self.assertEqual(kernel.matrix[1][1], 1)

    def test_get_blur_kernel(self):
        """Тест: получение ядра размытия"""
        kernel = get_kernel_by_name('blur')
        self.assertEqual(kernel.name, 'blur')

        total = sum(sum(row) for row in kernel.matrix)
        self.assertAlmostEqual(total, 1.0, places=5)

    def test_get_edge_kernel(self):
        """Тест: получение ядра обнаружения границ"""
        kernel = get_kernel_by_name('edge')
        self.assertEqual(kernel.name, 'edge')

    def test_get_sharpen_kernel(self):
        """Тест: получение ядра повышения резкости"""
        kernel = get_kernel_by_name('sharpen')
        self.assertEqual(kernel.name, 'sharpen')

    def test_get_unknown_kernel(self):
        """Тест: получение несуществующего ядра"""
        with self.assertRaises(ValueError):
            get_kernel_by_name('unknown_kernel')


class TestDesktopIntegration(unittest.TestCase):
    """Интеграционное тестирование полного цикла"""

    def test_full_pipeline(self):
        """Тест: полный цикл работы приложения"""

        input_image = [[50, 100, 150], [100, 150, 200], [150, 200, 250]]

        input_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        output_file = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False, encoding='utf-8')
        input_file.close()
        output_file.close()

        try:
            save_image_to_file(input_image, input_file.name)

            loaded_image = load_image_from_file(input_file.name)
            self.assertEqual(input_image, loaded_image)

            kernel = get_kernel_by_name('identity')
            convolution = ImageConvolution(kernel)
            result_image = convolution.apply(loaded_image)

            save_image_to_file(result_image, output_file.name)

            final_result = load_image_from_file(output_file.name)

            self.assertEqual(input_image, final_result)
        finally:

            os.remove(input_file.name)
            os.remove(output_file.name)


if __name__ == '__main__':

    print("Запуск тестирования десктоп-приложения «Свертка изображения»...\n")
    runner = unittest.TextTestRunner(stream=sys.stdout, verbosity=2)
    result = runner.run(unittest.TestLoader().loadTestsFromModule(sys.modules[__name__]))

    with open("desktop_test_protocol.txt", "w", encoding="utf-8") as f:
        f.write("=== ПРОТОКОЛ ТЕСТИРОВАНИЯ ДЕСКТОП-ПРИЛОЖЕНИЯ ===\n")
        f.write("Приложение: Свертка изображения\n")
        f.write("Тип тестирования: Тестирование десктоп-приложения\n\n")
        f.write(f"Всего выполнено тестов: {result.testsRun}\n")
        f.write(f"Успешно пройдено: {result.testsRun - len(result.failures) - len(result.errors)}\n")
        f.write(f"Ошибок выполнения (Errors): {len(result.errors)}\n")
        f.write(f"Провалов проверок (Failures): {len(result.failures)}\n")
        f.write("-" * 50 + "\n")
        if result.wasSuccessful():
            f.write("ИТОГ: Все тесты пройдены успешно. Приложение работает корректно.\n")
        else:
            f.write("ИТОГ: Обнаружены дефекты. Требуется доработка кода.\n")

    print("\nПротокол сохранён в файл 'desktop_test_protocol.txt'")