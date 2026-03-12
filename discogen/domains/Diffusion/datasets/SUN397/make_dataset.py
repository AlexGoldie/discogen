import datasets

def download_dataset(dest_loc: str = "./SUN397/data"):
    ds_dict = datasets.load_dataset("tanganke/sun397")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./SUN397/data"):
    dataset = datasets.load_from_disk(src_loc)

    return {"train": dataset["train"], "test": dataset["test"]}
