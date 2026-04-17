import datasets


def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("slegroux/tiny-imagenet-200-clean")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./TinyImageNet/data"):
    dataset = datasets.load_from_disk(src_loc)

    return {"train": dataset["train"], "test": dataset["test"]}
