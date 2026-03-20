import datasets


def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("benjamin-paine/imagenet-1k-64x64")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./ImageNet1k/data"):
    dataset = datasets.load_from_disk(src_loc)
    train = datasets.concatenate_datasets([dataset["train"], dataset["validation"]])

    return {"train": train, "test": dataset["test"]}
