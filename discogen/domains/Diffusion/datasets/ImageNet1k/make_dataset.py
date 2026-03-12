import datasets

def download_dataset(dest_loc: str = "./ImageNet1k/data"):
    ds_dict = datasets.load_dataset("ILSVRC/imagenet-1k")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./ImageNet1k/data"):
    dataset = datasets.load_from_disk(src_loc)

    return {"train": dataset["train"], "test": dataset["test"]}
