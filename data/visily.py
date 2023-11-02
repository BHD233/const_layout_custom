import json
from pathlib import Path

import torch
from torch_geometric.data import Data

from data.base import BaseDataset


def append_child(element, elements):
    if "children" in element.keys():
        for child in element["children"]:
            elements.append(child)
            elements = append_child(child, elements)
    return elements


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

    if element_type not in labels:
        return None

    return element_type


class Visily(BaseDataset):
    labels = [
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
    ]

    def __init__(self, split="train", transform=None):
        super().__init__("visily", split, transform)
        self.std_labels = [
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
        ]

    def download(self):
        super().download()

    def process(self):
        data_list = []
        raw_dir = Path(self.raw_dir) / "children_data"

        for json_path in sorted(raw_dir.glob("*.json")):
            with json_path.open() as f:
                ann = json.load(f)

            for parent in ann:
                W, H = float(parent["width"]), float(parent["height"])

                elements = parent["children"]
                N = len(elements)
                if N == 0 or 12 < N:
                    continue

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
                    l = normalize_type(element["type"], self.std_labels)
                    if l is not None:
                        labels.append(self.label2index[l])
                        boxes.append(b)

                boxes = torch.tensor(boxes, dtype=torch.float)
                labels = torch.tensor(labels, dtype=torch.long)

                data = Data(x=boxes, y=labels)
                data.attr = {
                    "name": json_path,
                    "width": W,
                    "height": H,
                    "filtered": True,
                    "has_canvas_element": False,
                }
                data_list.append(data)

        # shuffle with seed
        generator = torch.Generator().manual_seed(0)
        indices = torch.randperm(len(data_list), generator=generator)
        data_list = [data_list[i] for i in indices]

        # train 85% / val 5% / test 10%
        N = len(data_list)
        s = [int(N * 0.85), int(N * 0.90)]
        torch.save(self.collate(data_list[: s[0]]), self.processed_paths[0])
        torch.save(self.collate(data_list[s[0] : s[1]]), self.processed_paths[1])
        torch.save(self.collate(data_list[s[1] :]), self.processed_paths[2])
