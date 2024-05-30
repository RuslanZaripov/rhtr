import os
import multiprocessing
import json
import gzip
import pickle
import numpy as np
import cv2
import pyclipper
from shapely.geometry import Polygon
from scipy import ndimage
import pandas as pd
import torch
from torch.utils.data import Dataset

from src.segmentation.predictor import is_valid_polygon, rescale_contour
from src.segmentation.utils import get_filename, delete_and_create_dir
from src.segmentation.config import Config

print(f"{multiprocessing.cpu_count()=}")
semaphore = multiprocessing.Semaphore(multiprocessing.cpu_count() - 1)


def numbers2coords(list_of_numbers):
    """Convert list of numbers to list of tuple coords x, y."""
    bbox = [[list_of_numbers[i], list_of_numbers[i + 1]]
            for i in range(0, len(list_of_numbers), 2)]
    return np.array(bbox)


class SchoolSegmentationDataset(Dataset):
    """Image segmentation dataset."""

    def __init__(
            self,
            train_config: Config,
            json_annotations_path,
            image_root,
            save_dir_path,
            preprocessing=None,
            transform=None,
            df_path=None,
            keep=False
    ):
        """
        :param json_annotations_path: Path to the json file with annotations.
        :param image_root: Path to the folder with images.
        :param save_dir_path: Path to the folder where the processed data will be stored.
        :param preprocessing: Preprocessing function.
        :param transform: Transform function.
        :param df_path: Path to the dataframe with processed data.
        :param keep: If True, the processed data will be cached.
        """
        self.train_config = train_config

        self.image_root = image_root
        with open(json_annotations_path, 'r') as f:
            self.data = json.load(f)

        text_categories = ["pupil_text", "teacher_comment"]
        line_categories = ["text_line"]

        category_id2name = {category['name']: category['id'] for category in self.data['categories']}

        self.text_category_ids = [category_id2name[c] for c in text_categories]
        self.line_category_ids = [category_id2name[c] for c in line_categories]

        self.transform = transform
        self.preprocessing = preprocessing

        self.df_path = df_path
        if self.df_path is not None:
            self.processed_data_dir = os.path.dirname(self.df_path)
            self.df_filename = os.path.basename(self.df_path)

            self.target_dir = 'target'
            self.processed_images_dir = 'image'

            self.df = pd.read_csv(self.df_path)

            self.target_dir_path = f'{self.processed_data_dir}/{self.target_dir}'
            self.processed_image_dir_path = f'{self.processed_data_dir}/{self.processed_images_dir}'

        self.keep = keep
        if self.keep and df_path is None:
            root_ext = get_filename(json_annotations_path)

            self.processed_data_dir = f'{save_dir_path}/{root_ext}'
            self.target_dir = 'target'
            self.processed_images_dir = 'image'
            self.df_filename = f'df_{root_ext}.csv'

            self.df = pd.DataFrame(
                columns=['Image_Name', 'Processed_Image_Name', 'Target_Name'])

            self.target_dir_path = f'{self.processed_data_dir}/{self.target_dir}'
            self.processed_image_dir_path = f'{self.processed_data_dir}/{self.processed_images_dir}'

            delete_and_create_dir(self.target_dir_path)
            delete_and_create_dir(self.processed_image_dir_path)

    def __len__(self):
        return len(self.data['images'])

    def save_df(self):
        if self.keep or self.df_path is not None:
            self.df.to_csv(f'{self.processed_data_dir}/{self.df_filename}', index=False)

    def get_class_polygons(self, image_id, categories):
        """Get polygons from annotation for a specific image and category."""
        polygons = []
        for data_ann in self.data['annotations']:
            if (
                    data_ann['image_id'] == image_id
                    and data_ann['category_id'] in categories
                    and data_ann['segmentation']
            ):
                polygon = numbers2coords(data_ann['segmentation'][0])
                polygons.append(polygon)
        return polygons

    @staticmethod
    def add_shrinked_mask(i, segmentation, shrink_ratio, color):
        poly = Polygon(segmentation)
        pco = pyclipper.PyclipperOffset()
        pco.AddPath(segmentation, pyclipper.JT_ROUND, pyclipper.ET_CLOSEDPOLYGON)
        if not is_valid_polygon(poly):
            return False
        distance = int(poly.area * (1 - shrink_ratio ** 2) / poly.length)
        # polygon may split into several parts after shrink operation
        # https://stackoverflow.com/a/33902816
        shrinked_bboxes = pco.Execute(-distance)
        for shrinked_bbox in shrinked_bboxes:
            shrinked_bbox = np.array(shrinked_bbox)
            cv2.fillPoly(i, [shrinked_bbox], color)

    @staticmethod
    def add_border_mask(polygon, shrink_ratio):
        with semaphore:
            def distance_matrix(xs, ys, a, b):
                x1, y1 = a[0], a[1]
                x2, y2 = b[0], b[1]
                u1 = (((xs - x1) * (x2 - x1)) + ((ys - y1) * (y2 - y1)))
                u = u1 / (np.square(x1 - x2) + np.square(y1 - y2))
                u[u <= 0] = 2
                ix = x1 + u * (x2 - x1)
                iy = y1 + u * (y2 - y1)
                distance = np.sqrt(np.square(xs - ix) + np.square(ys - iy))
                distance2 = np.sqrt(np.fmin(np.square(xs - x1) + np.square(ys - y1),
                                            np.square(xs - x2) + np.square(ys - y2)))
                distance[u >= 1] = distance2[u >= 1]
                return distance

            polygon = np.array(polygon)
            assert polygon.ndim == 2, "The polygon must be a 2D array."
            assert polygon.shape[1] == 2, "The polygon must have 2 columns."
            poly = Polygon(polygon)
            if not is_valid_polygon(poly):
                return None
            distance = poly.area * (1 - np.power(shrink_ratio, 2)) / poly.length
            subject = [tuple(l) for l in polygon]
            padding = pyclipper.PyclipperOffset()
            padding.AddPath(subject, pyclipper.JT_ROUND,
                            pyclipper.ET_CLOSEDPOLYGON)
            padded_polygon = np.array(padding.Execute(distance)[0])

            xmin = padded_polygon[:, 0].min()
            xmax = padded_polygon[:, 0].max()
            ymin = padded_polygon[:, 1].min()
            ymax = padded_polygon[:, 1].max()
            width = xmax - xmin + 1
            height = ymax - ymin + 1

            polygon[:, 0] = polygon[:, 0] - xmin
            polygon[:, 1] = polygon[:, 1] - ymin
            xs = np.broadcast_to(
                np.linspace(0, width - 1, num=width).reshape(1, width),
                (height, width))
            ys = np.broadcast_to(
                np.linspace(0, height - 1, num=height).reshape(height, 1),
                (height, width))
            distance_map = np.zeros(
                (polygon.shape[0], height, width), dtype=np.float32)
            for i in range(polygon.shape[0]):
                j = (i + 1) % polygon.shape[0]
                absolute_distance = distance_matrix(
                    xs, ys, polygon[i], polygon[j])
                distance_map[i] = np.clip(absolute_distance / distance, 0, 1)
            distance_map = np.min(distance_map, axis=0)

            return xmin, xmax, ymin, ymax, height, width, distance_map

    # @time_it
    def get_watershed_map(self, polygons, image_shape, shrink_range):
        watershed_energy_map = np.zeros(
            (len(shrink_range), image_shape[0], image_shape[1]), dtype=np.uint8)

        for idx, polygon in enumerate(polygons):
            for k, shrink_ratio in enumerate(shrink_range):
                self.add_shrinked_mask(
                    watershed_energy_map[k], np.int32(polygon), shrink_ratio, [1])

        return watershed_energy_map

    # @time_it
    def get_border_map(self, polygons, image_shape, shrink_range):
        import multiprocessing

        canvas = np.zeros(
            (len(shrink_range), image_shape[0], image_shape[1]), dtype=np.float32)

        with multiprocessing.Pool(processes=(multiprocessing.cpu_count() - 1)) as pool:
            for k, shrink_ratio in enumerate(shrink_range):
                result = []
                for polygon in polygons:
                    result.append(
                        pool.apply_async(self.add_border_mask, (polygon, shrink_ratio))
                    )
                for r in result:
                    async_result = r.get()
                    if async_result is None:
                        print(f"polygon wasn't processed")
                        continue
                    xmin, xmax, ymin, ymax, height, width, distance_map = async_result
                    xmin_valid = min(max(0, xmin), canvas[k].shape[1] - 1)
                    xmax_valid = min(max(0, xmax), canvas[k].shape[1] - 1)
                    ymin_valid = min(max(0, ymin), canvas[k].shape[0] - 1)
                    ymax_valid = min(max(0, ymax), canvas[k].shape[0] - 1)
                    canvas[k][ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1] = np.fmax(
                        1 - distance_map[
                            ymin_valid - ymin:ymax_valid - ymax + height,
                            xmin_valid - xmin:xmax_valid - xmax + width],
                        canvas[k][ymin_valid:ymax_valid + 1, xmin_valid:xmax_valid + 1])

        canvas = (canvas > 0.5).astype(np.uint8)
        return canvas

    @staticmethod
    def get_distance_mask(polygons, image_shape):
        distance_mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)
        mask = np.zeros((image_shape[0], image_shape[1]), dtype=np.uint8)

        for idx, polygon in enumerate(polygons):
            mask = cv2.fillPoly(mask, [np.int32(polygon)], [1])

            bbox = cv2.boundingRect(polygon.astype(np.float32))
            x, y, w, h = bbox

            offset = 5
            x = max(0, x - offset)
            y = max(0, y - offset)
            w = min(w + 2 * offset, distance_mask.shape[1] - x)
            h = min(h + 2 * offset, distance_mask.shape[0] - y)

            try:
                y_h = y + h
                x_w = x + w
                edt = ndimage.distance_transform_edt(
                    mask[y:y_h, x:x_w],
                    return_indices=False,
                )
                distance_mask[y:y_h, x:x_w] = edt
            except Exception as err:
                print(err)

        # print(f"{np.min(distance_mask)}-{np.max(distance_mask)}")
        # import seaborn as sns
        # sns.heatmap(distance_mask, cmap='rainbow', cbar=True, xticklabels=False, yticklabels=False)
        # plt.show()

        distance_mask = distance_mask.astype(np.float32)
        distance_mask /= np.max(distance_mask)
        distance_mask = np.expand_dims(distance_mask, axis=0)

        return distance_mask

    @staticmethod
    def polyline2polygon(polyline, thickness):
        """Transform the polyline into a polygon by adding new points to create a thin polygon."""
        polygon = []

        for point in polyline:
            polygon.append([point[0] - int(thickness / 2), point[1] - int(thickness / 2)])

        for point in reversed(polyline):
            polygon.append([point[0] + int(thickness / 2), point[1] + int(thickness / 2)])

        return np.array(polygon)

    # @time_it
    def get_lines_map(self, polylines, image_shape, thickness=10):
        lines_map = np.zeros(
            (image_shape[0], image_shape[1]), dtype=np.uint8)

        for idx, p in enumerate(polylines):
            polygon = self.polyline2polygon(p, thickness)
            cv2.fillPoly(lines_map, [np.int32(polygon)], [1])

        return lines_map[np.newaxis, :]

    def create_mask(self, image_id, image_shape):
        polygons = self.get_class_polygons(image_id, self.text_category_ids)
        line_polygons = self.get_class_polygons(image_id, self.line_category_ids)

        def mask_factory(source):
            factory = {
                'binary': lambda r: self.get_watershed_map(polygons, image_shape, shrink_range=r),
                'lines': lambda th: self.get_lines_map(line_polygons, image_shape, thickness=th),
                'border_mask': lambda r: self.get_border_map(polygons, image_shape, shrink_range=r),
                'watershed': lambda r: self.get_watershed_map(polygons, image_shape, shrink_range=r)
            }
            return factory[source]

        masks: (np.ndarray,) = ()
        rbounds = [0]
        for key, value in self.train_config.get_masks().items():
            mask: np.ndarray = mask_factory(key)(value)
            rbounds.append(rbounds[-1] + mask.shape[0])
            masks += (mask,)

        target = np.concatenate(masks, axis=0)

        target = np.transpose(target, (1, 2, 0))

        return target, [polygons, line_polygons], rbounds

    def __getitem__(self, idx):
        data_img = self.data['images'][idx]

        img_name = str(data_img['file_name'])
        image_id = data_img['id']

        # print(f"{img_name=}")

        if self.keep and img_name in self.df['Image_Name'].values:
            target_name = self.df.loc[
                self.df['Image_Name'] == img_name, 'Target_Name'].values[0]
            processed_name = self.df.loc[
                self.df['Image_Name'] == img_name, 'Processed_Image_Name'].values[0]

            target_path = f'{self.processed_data_dir}/{self.target_dir}/{target_name}'
            image_path = f'{self.processed_data_dir}/{self.processed_images_dir}/{processed_name}'

            with gzip.open(target_path, 'rb') as f:
                loaded_dict = pickle.load(f)
            with gzip.open(image_path, 'rb') as f:
                processed_image = pickle.load(f)

            return processed_image, loaded_dict

        image_path = os.path.join(self.image_root, img_name)
        image = cv2.imread(image_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        if self.preprocessing is not None:
            image = self.preprocessing(image)

        target, polygons, rbounds = self.create_mask(image_id, image.shape)

        word_polygons = polygons[0]
        line_polygons = polygons[1]

        target_h, target_w = self.train_config.get_image('height'), self.train_config.get_image('width')

        # apply transforms
        if self.transform is not None:
            word_polygons = [rescale_contour(polygon,
                                             image.shape[0], image.shape[1],
                                             target_h, target_w)
                             for polygon in polygons[0]]

            line_polygons = [rescale_contour(polygon,
                                             image.shape[0], image.shape[1],
                                             target_h, target_w)
                             for polygon in polygons[1]]

            transformed = self.transform(image=image, mask=target)
            image, target = transformed['image'], transformed['mask']

        # convert to C, H, W
        image = image.transpose(2, 0, 1)
        target = target.transpose(2, 0, 1).astype(np.float64)

        target_dict = {}
        for idx, key in enumerate(self.train_config.get_masks().keys()):
            target_dict[key] = target[rbounds[idx]:rbounds[idx + 1]]
        target_dict['word_polygons'] = word_polygons
        target_dict['line_polygons'] = line_polygons

        if self.keep:
            target_filename = f'{get_filename(img_name)}_target.gz'
            image_filename = f'{get_filename(img_name)}_image.gz'

            target_path = f'{self.processed_data_dir}/{self.target_dir}/{target_filename}'
            image_path = f'{self.processed_data_dir}/{self.processed_images_dir}/{image_filename}'

            with gzip.open(target_path, 'wb') as f:
                pickle.dump(target_dict, f)
            with gzip.open(image_path, 'wb') as f:
                pickle.dump(image, f)

            data = {'Image_Name': [img_name],
                    'Processed_Image_Name': [image_filename],
                    'Target_Name': [target_filename]}
            self.df = pd.concat([self.df, pd.DataFrame(data)], ignore_index=True)

        return image, target_dict


def dynamic_length_collate(batch):
    max_line_polygons = max(len(item[1]['line_polygons']) for item in batch)
    max_word_polygons = max(len(item[1]['word_polygons']) for item in batch)

    for item in batch:
        line_polygons_pad_len = max_line_polygons - len(item[1]['line_polygons'])
        word_polygons_pad_len = max_word_polygons - len(item[1]['word_polygons'])

        item[1]['line_polygons'] += ([[]] * line_polygons_pad_len)
        item[1]['word_polygons'] += ([[]] * word_polygons_pad_len)

    dict_keys = batch[0][1].keys()

    images = torch.tensor(np.array([item[0] for item in batch]))

    targets = {key: [] for key in dict_keys}
    for item in batch:
        target_dict = item[1]
        for key, value in target_dict.items():
            targets[key].append(value)

    for key, value in targets.items():
        if 'polygons' in key: continue
        targets[key] = torch.tensor(np.array(value))

    return images, targets
