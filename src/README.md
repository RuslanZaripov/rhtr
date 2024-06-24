# Распознавание рукописного русского текста

Сегментация рукописного текста осуществляется моделью UNet + ResNet-50
c применением техники водораздела

Распознавание текста работает с помощью архитектуры TrOCR

# Примеры распознавания

[Видео демонстрация распознавания](/static/video.mp4)

![Пример распознавания 1](/static/1.png)

![Пример распознавания 2](/static/2.png)

# Описание структуры

1. [Весь процесс распознавания](/src/pipeline)
2. [Сегментации](/src/segmentation)
3. [Распознавание фрагментов](/src/ocr)

- Пример запуска и работы процесса распознавания находится в ноутбуке (pipeline/scripts/evaluate.ipynb)

# Источники

- Ronneberger O., Fischer P., Brox T. U-net: Convolutional networks for biomedical image segmentation
  – Springer International Publishing, 2015. – С. 234-241.
- Chaurasia A., Culurciello E. Linknet: Exploiting encoder representations for efficient semantic segmentation
  – IEEE, 2017. – С. 1-4.
- Bai M., Urtasun R. Deep watershed transform for instance segmentation – 2017. – С. 5221-5229
- Применяем Deep Watershed Transform. [Электронный ресурс].
  URL: https://habr.com/ru/articles/354040/
- Как написать пайплайн для чтения рукописного текста [Электронный ресурс].
  URL: https://habr.com/ru/companies/sberbank/articles/716796/
- Lin T. Y. et al. Focal loss for dense object detection – 2017. – С. 2980-2988
- Shi B., Bai X., Yao C. An end-to-end trainable neural network for image-based
  sequence recognition and its application to scene text recognition
  – 2016. – Т. 39. – №. 11. – С. 2298-2304.
- Li M. et al. Trocr: Transformer-based optical character recognition with
  pre-trained models – 2023. – Т. 37. – №. 11. – С. 13094-13102.
- Набор рукописных фрагментов текста на русском языке из платформы
  kaggle.com (Cyrillic Handwriting Dataset) [Электронный ресурс]. URL:
  https://www.kaggle.com/datasets/constantinwerner/cyrillic-handwriting-dataset
- Набор рукописных русских тетрадей [Электронный ресурс]. URL:
  https://huggingface.co/datasets/ai-forever/school_notebooks_RU
- Набор рукописных англоязычных тетрадей [Электронный ресурс]. URL:
  https://huggingface.co/datasets/ai-forever/school_notebooks_EN