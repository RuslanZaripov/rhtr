# Процесс распознавания

- Прочитать на другом язык [English](README.md)

## Описание структуры

- `abstract.py` - содержит абстрактные классы модулей распознавания для обобщения структуры кода
- `anglerestorer.py` - содержит код для вычисления угла поворота картинки
- `config.py` - содержит код необходимый для обработки файла конфигурации
- `linefinder.py` - содержит код необходимый для компоновки фрагментов текста
- `ocr_recognition.py` - содержит код для инициализации этапа распознавания
    - метод `recognizer_factory` отвечает за выбор модели на основе файлы конфигурации
- `word_segmentation.py` - содержит код для инициализации этапа сегментации
    - метод `segmentation_factory` отвечает за выбор модели на основе файла конфигурации
- `segm_postprocessing.py` - содержит код для инициализации этапа постобработки контуров
- `tr_ocr.py` - содержит код для модели распознавания натренированной библиотекой `transformers`
- `unet.py` - содержит код модели сегментации натренированной в модуле `segmentation`
- `utils.py` - содержит вспомогательный код процесса распознавания (визуализация, расчеты показателей)
- `scripts/pipeline_config.json` - пример файла конфигурации
- `scripts/evaluate.ipynb` - код запуска

Пример того, как выглядит файл конфигурации процесса распознавания:

- В этап сегментации `WordSegmentation` указывается название модели из метода `segmentation_factory`,
  путь к файлу с весами, путь к файлу конфигурации тренировочного процесса.
- После этапа сегментации указываются параметры модуля отработки результатов сегментации.
  С вариантами постобработки можно ознакомиться в файле `segm_postprocessing.py`.
  Ниже указан пример использования. Пишется название класса масок и модули обработки к нему.
  Важно соблюдать порядок.
- В этап распознавания `OpticalCharacterRecognition` указывается название модели из метода `recognizer_factory`,
  путь к директории или файлу с весами, название классов для распознавания.
- В этап компоновки указываются названия классов слов и линий.

```json
{
    "pipeline": {
        "WordSegmentation": {
            "model_name": "UNet",
            "model_path": "models/segmentation/${название_файла_весов}.onnx",
            "config_path": "src/segmentation/configs/${название_файла_конфигурации}.json"
        },
        "ContourPostprocessors": {
            "class2postprocessors": {
                "handwritten_text_mask": {
                    "BboxFromContour": {},
                    "UpscaleBbox": {
                        "upscale_bbox": [1, 1.5]
                    },
                    "CropByBbox": {}
                }
            }
        },
        "OpticalCharacterRecognition": {
            "model_name": "TrOCR",
            "model_path": "models/tr_ocr/",
            "ocr_classes": ["handwritten_text_mask"]
        },
        "LineFinder": {
            "line_classes": ["lines"],
            "text_classes": ["handwritten_text_mask"]
        }
    }
}
```

# Как я запускаю процесс распознавания?

- Пример работы написан в файле `src/pipeline/scripts/evaluate.ipynb`
- Скачать веса можно по ссылке `https://disk.yandex.ru/d/rxlpAgiTJYWrjA`
  (рекомендуется сделать такую же структуру папок как в ссылке выше).
  Корневая директория `/models` и папки `ocr/` и `segmentation` внутри.
- Кладем веса в папку `rhtr/models/segmentation` и указываем путь в конфиге `src/pipeline/scripts/pipeline_config.json`
- Лучше для запуска воспользоваться ноутбуком, открыв его на платформе kaggle.com
