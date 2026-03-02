import datasets


def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("nkirschi/oxford-flowers")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./OxfordFlowers/data"):
    dataset = datasets.load_from_disk(src_loc)

    return {"train": dataset["train"], "test": dataset["test"]}
