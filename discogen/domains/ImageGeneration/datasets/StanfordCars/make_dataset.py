import datasets


def download_dataset(dest_loc: str):
    ds_dict = datasets.load_dataset("tanganke/stanford_cars")
    ds_dict.save_to_disk(dest_loc)


def load_dataset(src_loc: str = "./StanfordCars/data"):
    from config import config
    dataset_dict = datasets.load_from_disk(src_loc)

    requested_train_splits = config["dataset"]["train_keys"]
    requested_test_splits = config["dataset"]["test_keys"]
    assert (set(requested_train_splits + requested_test_splits)).issubset(set(dataset_dict.keys())), (
        f"Some requested splits are not present in the dataset."
        f" Available splits: {list(dataset_dict.keys())}"
    )

    combined_train = datasets.concatenate_datasets([dataset_dict[k] for k in requested_train_splits])
    combined_test = datasets.concatenate_datasets([dataset_dict[k] for k in requested_test_splits])

    print("Dataset split finished: number of train samples =", len(combined_train), ", number of test samples =", len(combined_test))

    return {"train": combined_train, "test": combined_test}
