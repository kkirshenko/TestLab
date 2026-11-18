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
    """Класс для применения свертки к изображению"""

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