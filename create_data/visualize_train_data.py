import json
import random
import shutil
import numpy as np
from pathlib import Path
from datetime import datetime
from PIL import Image, ImageDraw
import seaborn as sns
import os


n_colors = 14
colors = sns.color_palette("husl", n_colors=n_colors)
colors = [tuple(map(lambda x: int(x * 255), c)) for c in colors]


std_labels = [
    "BUTTON",
    "CHART",
    "TABLE",
    "TEXT",
    "ICON",
    "TABBAR_MENU",
    "CONTAINER",
    "TEXTBOX",
    "CHECKBOX",
    "IMAGE",
    "HEADER_MENU",
    "SIDEBAR_MENU",
    "TAG",
    "SHAPE",
]


def normalize_type(element_type, labels):
    if "CHART" in element_type:
        return "CHART"

    if element_type == "BUTTON_GROUP":
        return "TAG"

    if "BUTTON" in element_type:
        return "BUTTON"

    if element_type in ["GROUP"]:
        return "CONTAINER"

    if element_type == "TEXTAREA":
        return "TEXTBOX"

    if element_type in ["ICON", "RATING", "AVATAR"]:
        return "ICON"

    if element_type == "RADIO":
        return "CHECKBOX"

    if element_type in ["TRIANGLE", "OVAL", "RECTANGLE", "DIVIDER", "SLIDER"]:
        return "SHAPE"

    if "SIDEBAR_MENU" in element_type:
        return "SIDEBAR_MENU"

    if element_type not in std_labels:
        return None

    return element_type


def convert_xywh_to_ltrb(bbox):
    xc, yc, w, h = bbox
    x1 = xc - w / 2
    y1 = yc - h / 2
    x2 = xc + w / 2
    y2 = yc + h / 2
    return [x1, y1, x2, y2]


def convert_layout_to_image(boxes, labels, canvas_size):
    H, W = canvas_size
    img = Image.new("RGB", (int(W), int(H)), color=(255, 255, 255))
    draw = ImageDraw.Draw(img, "RGBA")

    # draw from larger boxes
    area = [b[2] * b[3] for b in boxes]
    indices = sorted(range(len(area)), key=lambda i: area[i], reverse=True)

    for i in indices:
        bbox, color = boxes[i], colors[labels[i]]
        c_fill = color + (100,)
        x1, y1, x2, y2 = convert_xywh_to_ltrb(bbox)
        x1, x2 = x1 * (W - 1), x2 * (W - 1)
        y1, y2 = y1 * (H - 1), y2 * (H - 1)
        draw.rectangle([x1, y1, x2, y2], outline=color, fill=c_fill)
    return img


def main():
    i = 0
    for file_name in os.listdir("children_data"):
        if ".json" in file_name:
            with open(f"children_data/{file_name}") as f:
                objs = json.load(f)
                for parent in objs:
                    W, H = float(parent["width"]), float(parent["height"])

                    elements = parent["children"]

                    boxes = []
                    labels = []

                    for element in elements:
                        # bbox
                        if "data" in element:
                            x1 = element["data"]["position"]["x"]
                            y1 = element["data"]["position"]["y"]
                            width = (
                                element["data"]["width"]
                                if "width" in element["data"]
                                and element["data"]["width"] is not None
                                else 10
                            )
                            height = (
                                element["data"]["height"]
                                if "height" in element["data"]
                                and element["data"]["height"] is not None
                                else 10
                            )
                        else:
                            x1 = element["position"]["x"]
                            y1 = element["position"]["y"]
                            width = (
                                element["width"]
                                if "width" in element and element["width"] is not None
                                else 10
                            )
                            height = (
                                element["height"]
                                if "height" in element and element["height"] is not None
                                else 10
                            )

                        xc = x1 + width / 2.0
                        yc = y1 + height / 2.0
                        b = [xc / W, yc / H, width / W, height / H]

                        # label
                        l = normalize_type(element["type"], labels)
                        if l is not None:
                            labels.append(std_labels.index(l))
                            boxes.append(b)

                    img = convert_layout_to_image(boxes, labels, (H, W))

                    img.save(f"data_visual/{i}.png")
                    print(file_name, len(boxes))
                    i += 1


main()
