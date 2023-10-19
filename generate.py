import pickle
import argparse
from pathlib import Path
import json

import torch
from torch_geometric.data import DataLoader
from torch_geometric.utils import to_dense_batch

from util import set_seed, convert_layout_to_image
from data import get_dataset
from model.layoutganpp import Generator

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


def get_values_with_key(data, key, container_only=True):
    values = []
    if isinstance(data, dict):
        for k, v in data.items():
            if k == key:
                if (
                    container_only
                    and "type" in data
                    and data["type"] in ["CONTAINER", "TEMPLATE", "PAGE"]
                    and isinstance(v, list)
                    and len(v) > 0
                ):
                    values.append(
                        {
                            "width": data["data"]["width"],
                            "height": data["data"]["height"],
                            "children": v,
                        }
                    )

                # if isinstance(v, str):
                #     values.append(v)
            if isinstance(v, (dict, list)):
                print("Loop v", v)
                values.extend(get_values_with_key(v, key))
    elif isinstance(data, list):
        for item in data:
            values.extend(get_values_with_key(item, key))

    return values


def convert(file_name="data.json"):
    with open(file_name) as f:
        data = json.load(f)

    value = get_values_with_key(data, "children", container_only=True)[0]
    # print("value", value)
    W, H = float(value["width"]), float(value["height"])

    elements = value["children"]
    N = len(elements)
    # if N == 0 or 12 < N:
    #     continue

    boxes = []
    labels = []

    for element in elements:
        # bbox
        if "data" in element:
            x1 = element["data"]["position"]["x"]
            y1 = element["data"]["position"]["y"]
            width = (
                element["data"]["width"]
                if "width" in element["data"] and element["data"]["width"] is not None
                else 10
            )
            height = (
                element["data"]["height"]
                if "height" in element["data"] and element["data"]["height"] is not None
                else 10
            )

            xc = x1 + width / 2.0
            yc = y1 + height / 2.0
            b = [xc / W, yc / H, width / W, height / H]

            # label
            la = normalize_type(element["type"], labels)
            print("label", element["type"], la)
            if la is not None:
                labels.append(std_labels.index(la))
                boxes.append(b)

    boxes = torch.tensor(boxes, dtype=torch.float)
    labels = torch.tensor(labels, dtype=torch.long)

    return boxes, labels


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("ckpt_path", type=str, help="checkpoint path")
    parser.add_argument("--batch_size", type=int, default=1, help="batch size")
    parser.add_argument(
        "-o",
        "--out_path",
        type=str,
        default="output/generated_layouts.pkl",
        help="output pickle path",
    )
    parser.add_argument(
        "--num_save", type=int, default=0, help="number of layouts to save as images"
    )
    parser.add_argument("--seed", type=int, help="manual seed")
    args = parser.parse_args()

    if args.seed is not None:
        set_seed(args.seed)

    out_path = Path(args.out_path)
    out_dir = out_path.parent
    out_dir.mkdir(exist_ok=True, parents=True)

    # load checkpoint
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.ckpt_path, map_location=device)
    train_args = ckpt["args"]

    # load test dataset
    dataset = get_dataset(train_args["dataset"], "test")
    dataloader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        num_workers=4,
        pin_memory=True,
        shuffle=False,
    )
    num_label = dataset.num_classes

    # setup model and load state
    netG = (
        Generator(
            train_args["latent_size"],
            num_label,
            d_model=train_args["G_d_model"],
            nhead=train_args["G_nhead"],
            num_layers=train_args["G_num_layers"],
        )
        .eval()
        .to(device)
    )
    netG.load_state_dict(ckpt["netG"])

    results = []
    with torch.no_grad():
        for data in dataloader:
            boxes, labels = convert()
            data = data.to(device)
            # label, mask = to_dense_batch([labels], data.batch)
            # padding_mask = ~mask
            # z = torch.randn(
            #     label.size(0), label.size(1), train_args["latent_size"], device=device
            # )

            padding_mask = torch.tensor([[False for x in labels]])

            boxes = torch.unsqueeze(boxes, dim=0)
            labels = labels.unsqueeze(dim=0)
            print("label", labels)
            print("z", boxes)
            print("padding", padding_mask)

            bbox = netG(boxes, labels, padding_mask)

            print(bbox, padding_mask)

            for j in range(bbox.size(0)):
                mask_j = ~padding_mask[j]
                b = bbox[j][mask_j].cpu().numpy()
                l = labels[j][mask_j].cpu().numpy()

                if len(results) < args.num_save:
                    convert_layout_to_image(b, l, dataset.colors, (120, 80)).save(
                        out_dir / f"generated_{len(results)}.png"
                    )

                results.append((b, l))

            break

    # save results
    with out_path.open("wb") as fb:
        pickle.dump(results, fb)
    print("Generated layouts are saved at:", args.out_path)


if __name__ == "__main__":
    main()
