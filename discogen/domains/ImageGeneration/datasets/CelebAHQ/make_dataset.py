import datasets


def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("korexyz/celeba-hq-256x256")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./CelebAHQ/data"):
    dataset = datasets.load_from_disk(src_loc)

    return {"train": dataset["train"], "test": dataset["validation"]}
