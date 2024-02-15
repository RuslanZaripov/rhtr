# Segmentation library

# Splitting coco annotations into test and train sets

- [Code source](https://github.com/akarazniewicz/cocosplit)
- Source running command:

```bash
python cocosplit.py --having-annotations --multi-class -s 0.8 /path/to/your/coco_annotations.json train.json test.json
```

- Usage example on magazine dataset:

```bash
python .\src\segmentation\scripts\cocosplit.py --having-annotations --multi-class -s 0.8 .\data\raw\magazine\annotations.json .\data\raw\magazine\annotations_train.json .\data\raw\magazine\annotations_test.json
```

# Train segmentation model

## Prepate target masks from annotated polygons

- Source running command:

```bash
python scripts/prepare_dataset.py --config_path path/to/the/segm_config.json
```

- Usage example on magazine dataset:

```bash
python .\src\segmentation\scripts\prepare_dataset.py --config_path .\src\segmentation\scripts\segm_config.json
```

## Train segmentation model

- Source running command:

```bash
python scripts/train.py --config_path path/to/the/segm_config.json
```

- Usage example on magazine dataset:

```bash
python .\src\segmentation\scripts\train.py --config_path .\src\segmentation\scripts\segm_config.json
```
