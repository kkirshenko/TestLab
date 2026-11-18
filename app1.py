import argparse
import os


class Kernel:
    """Класс, представляющий ядро свертки (матрицу весов)."""

    def __init__(self, name: str, matrix: list):
        self.name = name
        self.matrix = matrix

        rows = len(self.matrix)
        for row in self.matrix:
            if len(row) != rows:
                raise ValueError("Матрица ядра должна быть квадратной")

        if rows % 2 == 0:
            raise ValueError("Размер ядра должен быть нечетным")


class ImageConvolution:
    """Класс для применения свертки к изображению."""

    def __init__(self, kernel: Kernel):
        self.kernel = kernel

    def apply(self, image: list) -> list:
        k_size = len(self.kernel.matrix)
        pad = k_size // 2
        height = len(image)
        width = len(image[0])

        result = [[0 for _ in range(width)] for _ in range(height)]

        for y in range(height):
            for x in range(width):
                pixel_sum = 0
                for ky in range(k_size):
                    for kx in range(k_size):
                        img_y = min(max(0, y + ky - pad), height - 1)
                        img_x = min(max(0, x + kx - pad), width - 1)
                        pixel_sum += image[img_y][img_x] * self.kernel.matrix[ky][kx]

                result[y][x] = max(0, min(255, int(pixel_sum)))

        return result


def load_image_from_file(filepath: str) -> list:
    """Загружает изображение из текстового файла."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Файл не найден: {filepath}")

    image = []
    with open(filepath, 'r', encoding='utf-8') as f:
        for line in f:
            line = line.strip()
            if line:
                row = [int(x) for x in line.split(',')]
                image.append(row)

    if not image:
        raise ValueError("Файл пуст или содержит некорректные данные")

    return image


def save_image_to_file(image: list, filepath: str):
    """Сохраняет изображение в текстовый файл."""
    with open(filepath, 'w', encoding='utf-8') as f:
        for row in image:
            f.write(','.join(map(str, row)) + '\n')


def get_kernel_by_name(name: str) -> Kernel:
    """Возвращает стандартное ядро по имени."""
    kernels = {
        'identity': [[0, 0, 0], [0, 1, 0], [0, 0, 0]],
        'blur': [[1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9], [1 / 9, 1 / 9, 1 / 9]],
        'edge': [[-1, -1, -1], [-1, 8, -1], [-1, -1, -1]],
        'sharpen': [[0, -1, 0], [-1, 5, -1], [0, -1, 0]]
    }

    if name not in kernels:
        raise ValueError(f"Неизвестное ядро: {name}. Доступные: {', '.join(kernels.keys())}")

    return Kernel(name, kernels[name])

def main():
    """Интерфейс командной строки."""
    parser = argparse.ArgumentParser(description='Свертка изображения')
    parser.add_argument('input', help='Входной файл изображения')
    parser.add_argument('output', help='Выходной файл изображения')
    parser.add_argument('--kernel', default='blur',
                        help='Имя ядра (identity, blur, edge, sharpen)')

    args = parser.parse_args()

    try:
        image = load_image_from_file(args.input)
        kernel = get_kernel_by_name(args.kernel)
        convolution = ImageConvolution(kernel)
        result = convolution.apply(image)
        save_image_to_file(result, args.output)
        print(f"Обработка завершена. Результат сохранен в {args.output}")
    except Exception as e:
        print(f"Ошибка: {e}")
        return 1

    return 0


if __name__ == '__main__':
    exit(main())