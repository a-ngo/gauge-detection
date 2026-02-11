import os

import cv2
import matplotlib.pyplot as plt


def plot_yolov8_bbox(image_path: str, label_path: str) -> None:
    """
    Displays an image with YOLOv8 bounding boxes drawn from a label file.

    Args:
        image_path (str): Path to the input image file.
        label_path (str): Path to the YOLOv8 label file containing bounding box annotations.

    Description:
        This function reads an image and its corresponding YOLOv8 label file, draws bounding boxes on the image
        based on the annotations, and displays the result using matplotlib. If the label file does not exist,
        the image is displayed with a "No label found" title.
    """
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    h, w, _ = img.shape
    if not os.path.exists(label_path):
        plt.imshow(img)
        plt.title("No label found")
        plt.axis("off")
        return
    with open(label_path, "r") as f:
        for line in f:
            parts = line.strip().split()
            if len(parts) != 5:
                continue
            _, x_center, y_center, box_w, box_h = map(float, parts)
            x_center, y_center, box_w, box_h = (
                x_center * w,
                y_center * h,
                box_w * w,
                box_h * h,
            )
            x1 = int(x_center - box_w / 2)
            y1 = int(y_center - box_h / 2)
            x2 = int(x_center + box_w / 2)
            y2 = int(y_center + box_h / 2)
            img = cv2.rectangle(img, (x1, y1), (x2, y2), (255, 0, 0), 2)
    plt.imshow(img)
    plt.axis("off")
