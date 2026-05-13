import re

# Define placeholders for dataset paths
CAMBRIAN_737K = {
    "annotation_path": "PATH_TO_CAMBRIAN_737K_ANNOTATION",
    "data_path": "",
}

# VisualWebInstruct (HF: TIGER-Lab/VisualWebInstruct, config=conversation)
# Expected local layout (materialized from HF):
# - /data02/home/philip.yang/datasets/visualwebinstruct/train.jsonl
# - /data02/home/philip.yang/datasets/visualwebinstruct/val.jsonl
# - /data02/home/philip.yang/datasets/visualwebinstruct/images/...
VISUALWEBINSTRUCT_TRAIN = {
    "annotation_path": "/data02/home/philip.yang/datasets/visualwebinstruct/train.jsonl",
    "data_path": "/data02/home/philip.yang/datasets/visualwebinstruct",
}

VISUALWEBINSTRUCT_VAL = {
    "annotation_path": "/data02/home/philip.yang/datasets/visualwebinstruct/val.jsonl",
    "data_path": "/data02/home/philip.yang/datasets/visualwebinstruct",
}

CAMBRIAN_737K_PACK = {
    "annotation_path": f"PATH_TO_CAMBRIAN_737K_ANNOTATION_PACKED",
    "data_path": f"",
}

MP_DOC = {
    "annotation_path": "PATH_TO_MP_DOC_ANNOTATION",
    "data_path": "PATH_TO_MP_DOC_DATA",
}

CLEVR_MC = {
    "annotation_path": "PATH_TO_CLEVR_MC_ANNOTATION",
    "data_path": "PATH_TO_CLEVR_MC_DATA",
}

VIDEOCHATGPT = {
    "annotation_path": "PATH_TO_VIDEOCHATGPT_ANNOTATION",
    "data_path": "PATH_TO_VIDEOCHATGPT_DATA",
}

DEMO_SINGLE_IMAGES = {
    "annotation_path": "demo/single_images.json",
    "data_path": "qwen-vl-finetune",
}

data_dict = {
    "cambrian_737k": CAMBRIAN_737K,
    "cambrian_737k_pack": CAMBRIAN_737K_PACK,
    "visualwebinstruct_train": VISUALWEBINSTRUCT_TRAIN,
    "visualwebinstruct_val": VISUALWEBINSTRUCT_VAL,
    "mp_doc": MP_DOC,
    "clevr_mc": CLEVR_MC,
    "videochatgpt": VIDEOCHATGPT,
    "demo_single_images": DEMO_SINGLE_IMAGES,
}


def parse_sampling_rate(dataset_name):
    match = re.search(r"%(\d+)$", dataset_name)
    if match:
        return int(match.group(1)) / 100.0
    return 1.0


def data_list(dataset_names):
    config_list = []
    for dataset_name in dataset_names:
        sampling_rate = parse_sampling_rate(dataset_name)
        dataset_name = re.sub(r"%(\d+)$", "", dataset_name)
        if dataset_name in data_dict.keys():
            config = data_dict[dataset_name].copy()
            config["sampling_rate"] = sampling_rate
            config_list.append(config)
        else:
            raise ValueError(f"do not find {dataset_name}")
    return config_list


if __name__ == "__main__":
    dataset_names = ["cambrian_737k"]
    configs = data_list(dataset_names)
    for config in configs:
        print(config)
