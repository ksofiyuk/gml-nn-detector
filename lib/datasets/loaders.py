# --------------------------------------------------------
# Deep CNN Detector with Pretrained Models
# Copyright (c) 2016 Graphics & Media Lab
# Licensed under The MIT License [see LICENSE for details]
# Written by Konstantin Sofiyuk
# --------------------------------------------------------
import json
import os
from datasets.image_sample import ImageFileSampleCV


def load_bboxes_dataset_with_json_marking(dataset_path: str,
                                          marking_filename: str,
                                          max_size: int, scales: list) -> list:
    """Данная функция загружает набор данных, которые представлены совокупностью
    изображений и размеченными объектами.

    Структура директории с разметкой:
        dataset_path/marking_filename: разметка всех изображений
        dataset_path/imgs: директория, содержащая файлы изображений

    Структура файла разметки:
        {
            "image01.jpg": [
                {
                    x: координата левого верхнего угла по горизонтали,
                    y: координата левого верхнего угла по вертикали,
                    w: ширина объекта,
                    h: высота объекта,
                    class: номер класса объекта, может отсутствовать (default: 1),
                        0 - класс фона
                    ignore: игнорировать или нет объект (Boolean),
                        этот параметр может отсутствовать (default: False)
                }
            ], ...
        }
    "image01.jpg" - относительный путь к изображению по отношению к dataset_path/imgs.

    Args:
        dataset_path (str): путь к директории с разметкой и изображениями
        marking_filename (str): имя файла разметки
        max_size (int): при прогоне сети по этому изображению, размер наибольшей стороны
                        не может превосходить этого значения
        scales (list):  на этом изображении алгоритм будет применён на нескольких масштабах,
                        которые соответствуют изображениям с длиной наименьшей стороны из scales

    Returns:
        list: Список объектов типа ImageFileSampleCV
    """

    marking_path = os.path.join(dataset_path, marking_filename)

    with open(marking_path, 'r') as f:
        marking = json.load(f)

    samples = []
    for image_name, image_marking in sorted(marking.items()):
        image_path = os.path.join(dataset_path, 'imgs', image_name)

        for obj in image_marking:
            if 'ignore' not in obj:
                obj['ignore'] = False
            if 'class' not in obj:
                obj['class'] = 1

        image_sample = ImageFileSampleCV(image_path, image_marking,
                                         max_size, scales)
        samples.append(image_sample)

    return samples


def load_images_from_directory_without_marking(
        images_path: str, max_size: int, scales: list) -> list:
    """Загружает все изображения в форматах *.jpg,*.jpeg,*.png
    из указанной директории без разметки.

    Данная функция полезна для подготовки данных для тестирования на них детектора.

    Args:
        images_path: путь к папке, содержащей изображения
    Returns:
        list: Список объектов типа ImageFileSampleCV
    """

    files = [os.path.join(images_path, file_name)
             for file_name in os.listdir(images_path)]
    files = filter(os.path.isfile, files)
    images_files = sorted(filter(lambda x: x.endswith(('.jpg', '.png', '.jpeg')), files))

    return [ImageFileSampleCV(image_path, [], max_size, scales) for image_path in images_files]