import datasets
import numpy as np

def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("uoft-cs/cifar10")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./CIFAR10LT/data"):
    from config import config

    dataset = datasets.load_from_disk(src_loc)

    imbalance_factor = config["dataset"]["imbalance_factor"]
    num_train_samples = len(dataset["train"])
    img_max = num_train_samples / config["dataset"]["num_classes"]

    img_num_per_cls = []
    # Decide indices for each class
    for cls_idx in range(config["dataset"]["num_classes"]):
        num = img_max * (imbalance_factor**(cls_idx / (config["dataset"]["num_classes"] - 1.0)))
        img_num_per_cls.append(int(num))

    targets = dataset["train"][config["dataset"]["label_key"]]
    new_indices = []
    targets_np = np.array(targets, dtype=np.int64)
    classes = np.unique(targets_np)
    num_per_cls_dict = dict()
    for the_class, the_img_num in zip(classes, img_num_per_cls):
        num_per_cls_dict[the_class] = the_img_num
        idx = np.where(targets_np == the_class)[0]
        np.random.shuffle(idx)
        selec_idx = idx[:the_img_num]
        new_indices.extend(selec_idx.tolist())

    print("Dataset split finished: number of train samples =", len(new_indices), ", number of test samples =", len(dataset["test"]))

    return {"train": dataset["train"].select(new_indices), "test": dataset["test"]}
