import numpy as np
from shapely.ops import nearest_points
from shapely.geometry import Polygon
from sklearn.cluster import KMeans
import cv2
import matplotlib.pyplot as plt


def contour2shapely(contour):
    """Convert contour (list of [x, y]) to the shapley.Polygon."""
    if len(contour) < 3:
        return None
    polygon = Polygon(contour)
    if not polygon.is_valid:
        polygon = polygon.buffer(0)
    return polygon


def add_polygon_center(predictions):
    """
    Add center coords for each polygon in the predictions-dict.
    """
    for idx, prediction in enumerate(predictions['predictions']):
        contour = prediction['rotated_polygon'] if 'rotated_polygon' in prediction.keys() else prediction['polygon']
        # compute the center of the contour
        array = np.array(contour)
        assert len(array.shape) == 2, "not a 2d points array"
        M = cv2.moments(array)
        if M["m00"] == 0:
            cX = int(np.mean(array[..., 0]))
            cY = int(np.mean(array[..., 1]))
        else:
            cX = int(M['m10'] / M['m00'])
            cY = int(M['m01'] / M['m00'])
        predictions['predictions'][idx]['polygon_center'] = [cX, cY]


def is_pages_swapped(cluster_centers):
    """
    Check if page X clusters not sorted.
    Args:
        cluster_centers: list of 2d coordinates
    """
    return cluster_centers[0][0] > cluster_centers[1][0]


def has_two_pages(cluster_centers, img_w, max_diff):
    """
    Check if there are two pages on the image by comparing distance between K-means clusters of the image lines.
    """
    center1 = cluster_centers[0][0]
    center2 = cluster_centers[1][0]
    diff_ratio = abs(center2 - center1) / img_w
    return diff_ratio >= max_diff


def add_page_idx_for_lines(predictions, line_class_names, img_w, pages_clust_dist=.15):
    """
    Add page indexes for each contour in the pred_img-dict.
    Page is predicted using K-Means via line polygons.
    """
    x_coords = []
    indexes = []
    for idx, prediction in enumerate(predictions['predictions']):
        if prediction['class_name'] in line_class_names:
            contour_center = prediction['polygon_center']
            x_coords.append(contour_center[0])
            indexes.append(idx)

    page_indexes = [0 for _ in range(len(indexes))]
    if len(x_coords) >= 2:
        X = np.array(x_coords).reshape(-1, 1)
        kmeans = KMeans(n_clusters=2, random_state=0).fit(X)
        y_kmeans = kmeans.predict(X)

        # plt.scatter(X[:, 0], np.zeros_like(X[:, 0]), c=y_kmeans, s=50, cmap='viridis')
        # plt.savefig(f'images/lines_clustering.png')
        # plt.clf()

        if has_two_pages(kmeans.cluster_centers_, img_w, pages_clust_dist):
            page_indexes = [1 - int(page)
                            if is_pages_swapped(kmeans.cluster_centers_)
                            else int(page)
                            for page in kmeans.labels_]

    for idx, page_idx in zip(indexes, page_indexes):
        predictions['predictions'][idx]['page_idx'] = page_idx


def add_line_idx_for_lines(pred_img, line_class_names):
    """
    Add line indexes for line contours in the pred_img-dict.
    Line index is calculated by sorting line contours by their y-mean coords.
    """
    for page_idx in get_page_indexes(pred_img):
        y_means = []
        indexes = []
        for idx, prediction in enumerate(pred_img['predictions']):
            if (prediction['class_name'] in line_class_names
                    and prediction['page_idx'] == page_idx):
                y_means.append(prediction['polygon_center'][1])
                indexes.append(idx)

        sorted_indexes_y_means = sorted(
            zip(indexes, y_means), key=lambda x: x[1], reverse=False)

        for line_idx, (idx, y_mean) in enumerate(sorted_indexes_y_means):
            pred_img['predictions'][idx]['line_idx'] = line_idx


def get_polygons_distance(polygon1, polygon2):
    """
    Get distance between two polygons.
    """
    if polygon1 is None or polygon2 is None: return None
    return polygon1.distance(polygon2)


def get_idx_of_line_closest_to_word(word_contour, pred_img, line_class_names):
    """
    Get the index of the line closest to the input word contour.

    Args:
        word_contour (list of [x, y] coords): The contour of the word.
        pred_img (dict): The dictionary with predictions.
        line_class_names (list): The list of line class names.
    """
    indexes = []
    line_shapelys = []
    for idx, prediction in enumerate(pred_img['predictions']):
        if prediction['class_name'] in line_class_names:
            indexes.append(idx)
            line_shapely = contour2shapely(prediction['polygon'])
            line_shapelys.append(line_shapely)

    min_polygon_distance = np.inf
    idx_of_line = None
    word_shapely = contour2shapely(word_contour)

    for idx, line_shapely in zip(indexes, line_shapelys):
        polygons_distance = get_polygons_distance(line_shapely, word_shapely)
        if polygons_distance is not None:
            if polygons_distance == 0:
                return 0, idx
            elif polygons_distance < min_polygon_distance:
                min_polygon_distance = polygons_distance
                idx_of_line = idx

    return min_polygon_distance, idx_of_line


def add_line_idx_for_words(predictions, line_class_names, word_class_names):
    """
    Add line indexes for each word polygon in the predictions-dict.
    The word contour must intersect with the line contour to determine the
    line index for the word.
    """
    for prediction in predictions['predictions']:
        if prediction['class_name'] in word_class_names:

            dist, idx = get_idx_of_line_closest_to_word(
                prediction['polygon'], predictions, line_class_names)

            if idx is not None:
                prediction['page_idx'] = predictions['predictions'][idx]['page_idx']
                prediction['line_idx'] = predictions['predictions'][idx]['line_idx']


def add_column_idx_for_words(predictions, word_class_names):
    """
    Add column indexes for word contours in the predictions-dict.
    Column index is calculated by sorting word contours from one line
    by their x-mean coords.
    """
    for page_idx in get_page_indexes(predictions):
        for line_idx in get_line_indexes(predictions, page_idx):

            indexes = []
            x_means = []
            for idx, prediction in enumerate(predictions['predictions']):
                if (prediction.get('page_idx') == page_idx
                        and prediction.get('line_idx') == line_idx
                        and prediction['class_name'] in word_class_names):
                    indexes.append(idx)
                    x_means.append(prediction['polygon_center'][0])

            sorted_indexes_x_means = sorted(
                zip(indexes, x_means), key=lambda x: x[1], reverse=False)

            for column_idx, (idx, x_mean) in enumerate(sorted_indexes_x_means):
                predictions['predictions'][idx]['column_idx'] = column_idx


def get_page_indexes(predictions):
    """
    Get list of sorted unique page indexes from predictions-dict.
    """
    unique_page_indexes = set()
    for prediction in predictions['predictions']:
        page_idx = prediction.get('page_idx')
        if page_idx is not None:
            unique_page_indexes.add(page_idx)
    return sorted(list(unique_page_indexes), reverse=False)


def get_line_indexes(predictions, page_idx):
    """
    Get list of sorted unique line indexes from a given page from predictions-dict.
    """
    unique_line_indexes = set()
    for prediction in predictions['predictions']:
        if prediction.get('page_idx') == page_idx:
            line_idx = prediction.get('line_idx')
            if line_idx is not None:
                unique_line_indexes.add(line_idx)
    return sorted(list(unique_line_indexes), reverse=False)


def get_column_indexes(pred_img, page_idx, line_idx):
    """
    Get list of sorted unique column indexes from a given page and line from predictions-dict.
    """
    unique_column_indexes = set()
    for prediction in pred_img['predictions']:
        if (prediction.get('page_idx') == page_idx
                and prediction.get('line_idx') == line_idx):
            column_idx = prediction.get('column_idx')
            if column_idx is not None:
                unique_column_indexes.add(column_idx)
    return sorted(list(unique_column_indexes), reverse=False)


def add_word_indexes(predictions, word_class_names):
    """
    Add positional indexes for word contours.
    Using this index structured text can be extracted from the prediction.
    """
    word_idx = 0
    for page_idx in get_page_indexes(predictions):
        for line_idx in get_line_indexes(predictions, page_idx):
            for column_idx in get_column_indexes(predictions, page_idx, line_idx):
                for prediction in predictions['predictions']:
                    if (prediction.get('page_idx') == page_idx
                            and prediction.get('line_idx') == line_idx
                            and prediction.get('column_idx') == column_idx
                            and prediction['class_name'] in word_class_names):
                        prediction['word_idx'] = word_idx
                        word_idx += 1


def simple_ordering(pred_img):
    centroids = [(idx, prediction['polygon_center'])
                 for idx, prediction in enumerate(pred_img['predictions'])]

    sort = sorted(centroids, key=lambda p: p[1][0] + (p[1][1] * 10))
    for word_idx, polygon in enumerate(sort):
        pred_img['predictions'][polygon[0]]['word_idx'] = word_idx


def complex_ordering(pred_img):
    centroids = [(idx, prediction['polygon_center'])
                 for idx, prediction in enumerate(pred_img['predictions'])]

    sorted_centroids = []
    points_to_sort = centroids.copy()
    while len(points_to_sort) > 0:
        upper_left = sorted(points_to_sort, key=lambda pt: pt[1][0] + pt[1][1])[0]
        upper_right = sorted(points_to_sort, key=lambda pt: pt[1][0] - pt[1][1])[-1]

        upper_left = np.array([upper_left[1][0], upper_left[1][1], 0])
        upper_right = np.array([upper_right[1][0], upper_right[1][1], 0])

        row_points = []
        remaining_points = []
        for point in points_to_sort:
            p = np.array([point[1][0], point[1][1], 0])

            dist = (np.linalg.norm(
                np.cross(
                    np.subtract(p, upper_left),
                    np.subtract(upper_right, upper_left))) / np.linalg.norm(upper_right))

            if dist < 50:
                row_points.append(point)
            else:
                remaining_points.append(point)

        sorted_centroids.extend(sorted(row_points, key=lambda h: h[0]))
        points_to_sort = remaining_points

    for word_idx, centroid in enumerate(sorted_centroids):
        pred_img['predictions'][centroid[0]]['word_idx'] = word_idx


def visualize_ordering(image, pred_img, classes, idx_name, filename=None):
    image_copy = image.copy()

    for prediction in pred_img['predictions']:
        if prediction['class_name'] in classes:
            contour = prediction['rotated_polygon'] if 'rotated_polygon' in prediction.keys() else prediction['polygon']
            polygon = [tuple(point)
                       for point in contour]
            polygon_np = np.array(polygon, np.int32)
            polygon_np = polygon_np.reshape((-1, 1, 2))
            cv2.polylines(image_copy, [polygon_np],
                          isClosed=True,
                          color=(255, 0, 0),
                          thickness=2)

    plt.figure(figsize=(10, 10))
    plt.axis('off')
    plt.imshow(image_copy)

    for prediction in pred_img['predictions']:
        if prediction['class_name'] in classes:
            plt.scatter(prediction['polygon_center'][0], prediction['polygon_center'][1], color='red', marker='o')
            idx = str(prediction[idx_name]) if idx_name in prediction.keys() else ''
            plt.annotate(
                f"{idx}",
                (prediction['polygon_center'][0], prediction['polygon_center'][1]),
                fontsize=7)

    plt.savefig(f'images/word_ordering_{idx_name}.png' if filename is None else filename)
    plt.clf()


class LineFinder:
    """
    Heuristic methods to define indexes of rows, columns and pages for
    polygons on the image.

    Args:
        pages_clust_dist (float): Relative (to image width) distance between two
            clusters of lines' polygons to consider that image has two pages.
    """

    def __init__(self, args, pages_clust_dist=0.25):
        self.line_classes = args['line_classes']
        self.text_classes = args['text_classes']
        self.pages_clust_dist = pages_clust_dist

    def __call__(self, image, pred_img):
        _, img_w = image.shape[:2]
        add_polygon_center(pred_img)

        # simple_ordering(pred_img)
        # complex_ordering(pred_img)

        add_page_idx_for_lines(pred_img, self.line_classes, img_w, self.pages_clust_dist)

        # visualize_ordering(image, pred_img, self.line_classes, 'page_idx')

        add_line_idx_for_lines(pred_img, self.line_classes)

        # visualize_ordering(image, pred_img, self.line_classes, 'line_idx')

        add_line_idx_for_words(pred_img, self.line_classes, self.text_classes)

        # visualize_ordering(image, pred_img, self.text_classes, 'line_idx', 'images/word_ordering_line_idx_1.png')

        add_column_idx_for_words(pred_img, self.text_classes)

        # visualize_ordering(image, pred_img, self.text_classes, 'column_idx')

        add_word_indexes(pred_img, self.text_classes)

        # visualize_ordering(image, pred_img, self.text_classes, 'word_idx')

        return image, pred_img
