import json


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

                if isinstance(v, str):
                    values.append(v)
            if isinstance(v, (dict, list)):
                values.extend(get_values_with_key(v, key))
    elif isinstance(data, list):
        for item in data:
            values.extend(get_values_with_key(item, key))

    return values


with open("pattern_list.json") as f:
    patterns = json.load(f)

with open("template_list.json") as f:
    templates = json.load(f)

label = []
count9 = 0
count = 0
for c in patterns + templates:
    a = get_values_with_key(c, "children")

    count += len(a)

    count9 += len([x for x in a if len(x["children"]) < 12])

    with open(f"children_data/{c['id']}.json", "w") as f:
        f.write(json.dumps(a, indent=4))

    label.extend(get_values_with_key(c, "type", container_only=False))

with open(f"label.json", "w") as f:
    f.write(str(list(set(label))))
# with open(f"label.json", "w") as f:
#     f.write(json.dumps(list(set(label)), indent=4))

print(count9, count)
