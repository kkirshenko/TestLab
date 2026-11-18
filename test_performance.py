import time
from app import Kernel, ImageConvolution


def generate_test_image(size: int) -> list:
    """
    Генерирует тестовое изображение заданного размера.
    Каждый пиксель заполняется значением 128 (серый цвет).
    """
    return [[128 for _ in range(size)] for _ in range(size)]


def measure_convolution_time(image_size: int, kernel: Kernel) -> float:
    """
    Замеряет время выполнения свёртки на изображении заданного размера.
    """
    image = generate_test_image(image_size)
    convolution = ImageConvolution(kernel)

    start_time = time.perf_counter()  # Точный замер времени
    convolution.apply(image)
    end_time = time.perf_counter()

    return end_time - start_time


def main():
    # Стандартное ядро размытия 3x3
    blur_kernel = Kernel("blur_3x3", [
        [1, 1, 1],
        [1, 1, 1],
        [1, 1, 1]
    ])

    # Размеры изображений для тестирования (от малого к большому)
    image_sizes = [10, 50, 100, 200, 300, 400, 500]

    print("=== Тестирование производительности приложения «Свертка изображения» ===\n")
    print(f"{'Размер изображения':<25} {'Время выполнения (сек)':<25}")
    print("-" * 50)

    results = []
    for size in image_sizes:
        execution_time = measure_convolution_time(size, blur_kernel)
        results.append((size, execution_time))
        print(f"{size}x{size:<20} {execution_time:.6f}")

    # Размер, при котором время превышает 1 секунду (стресс-тест)
    stress_limit_size = None
    for size, time_sec in results:
        if time_sec > 1.0:
            stress_limit_size = size
            break

    print("\n--- Результаты стресс-тестирования ---")
    if stress_limit_size:
        print(f"При размере изображения {stress_limit_size}x{stress_limit_size} "
              f"время выполнения превышает 1 секунду.")
        print("Это считается пределом приемлемой производительности для интерактивной работы.")
    else:
        print("На всех протестированных размерах время выполнения менее 1 секунды.")

    with open("performance_protocol.txt", "w", encoding="utf-8") as f:
        f.write("=== ПРОТОКОЛ НЕФУНКЦИОНАЛЬНОГО ТЕСТИРОВАНИЯ ===\n")
        f.write("Приложение: Свертка изображения\n")
        f.write("Вид тестирования: Тестирование производительности\n\n")
        f.write("Параметры теста:\n")
        f.write("  - Ядро: blur_3x3 (размер 3x3)\n")
        f.write("  - Значение пикселей: 128 (серый)\n\n")
        f.write("Результаты замеров:\n")
        for size, time_sec in results:
            f.write(f"  Размер {size}x{size}: {time_sec:.6f} сек\n")
        f.write("\nВывод стресс-теста:\n")
        if stress_limit_size:
            f.write(f"  Предел производительности достигнут при размере {stress_limit_size}x{stress_limit_size}.\n")
        else:
            f.write("  Предел производительности не достигнут на протестированных размерах.\n")

    print("\nПротокол сохранён в файл 'performance_protocol.txt'")


if __name__ == '__main__':
    main()