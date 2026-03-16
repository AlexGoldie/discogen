import torch
from torch.utils.data import Dataset
from data.utils import (
    load_hf_dataset,
    add_dataset_index,
    preprocess_pretraining_instance,
)


class CompletionDataset(Dataset):
    def __init__(
        self,
        hf_args,
        template_args,
        tokenizer,
        prefix_key="prompt",
        text_key="text",
        max_length=2048,
        predict_with_generate=False,
        insert_space=False,
    ):
        super(CompletionDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.data = load_hf_dataset(**hf_args)
        self.data = add_dataset_index(self.data)
        # if either key does not exist in dataset, it is taken as ""
        self.prefix_key = prefix_key
        self.text_key = text_key
        self.predict_with_generate = predict_with_generate
        self.insert_space = insert_space

    def __len__(self):
        return len(self.data)

    def _process_sample(self, prefix, text_content, index=-1):
        tokenized_data = preprocess_pretraining_instance(
            self.tokenizer,
            prefix,
            text_content,
            self.max_length,
            self.predict_with_generate,
            self.insert_space,
        )
        item_dct = {
            "input_ids": tokenized_data["input_ids"],
            "labels": tokenized_data["labels"],
            "attention_mask": tokenized_data["attention_mask"],
        }
        if index != -1:
            item_dct["index"] = index
        return item_dct

    def __getitem__(self, idx):
        pref = self.data[idx].get(self.prefix_key, "")
        text_content = self.data[idx].get(self.text_key, "")
        index = self.data[idx]["index"]
        item = self._process_sample(pref, text_content, index)
        return item

class PretrainingDataset(Dataset):
    def __init__(
        self, hf_args, template_args, tokenizer, text_key="text", max_length=2048
    ):
        super(PretrainingDataset, self).__init__()
        self.tokenizer = tokenizer
        self.max_length = max_length

        # Load the raw dataset
        raw_dataset = load_hf_dataset(**hf_args)

        # Define the tokenization and chunking function
        def tokenize_and_chunk(examples):
            # 1. Tokenize all the texts. Add an EOS token between documents
            # to separate them.
            eos_token = tokenizer.eos_token or "<|endoftext|>"
            texts = [doc + eos_token for doc in examples[text_key] if doc is not None]

            # Tokenize, but don't add special tokens (we're in the middle of text)
            tokenized_inputs = tokenizer(texts, add_special_tokens=False)

            # 2. Concatenate all token lists into one big list
            concatenated_examples = {k: sum(tokenized_inputs[k], []) for k in tokenized_inputs.keys()}
            total_length = len(concatenated_examples[list(tokenized_inputs.keys())[0]])

            # 3. Split the big list into chunks of max_length
            # We drop the last chunk if it's smaller than max_length
            total_length = (total_length // max_length) * max_length

            result = {
                k: [t[i : i + max_length] for i in range(0, total_length, max_length)]
                for k, t in concatenated_examples.items()
            }

            # 4. Create the labels (which are just the input_ids)
            result["labels"] = result["input_ids"].copy()
            return result

        # 5. Apply this function to the dataset using .map()
        self.data = raw_dataset.map(
            tokenize_and_chunk,
            batched=True,
            remove_columns=raw_dataset.column_names, # Remove old text columns
        )

        print(f"--- PretrainingDataset: Created {len(self.data)} chunks of size {max_length}. ---")

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        # Data is already tokenized and preprocessed
        # We just need to convert to tensors
        item = self.data[idx]
        return {
            "input_ids": torch.tensor(item["input_ids"]),
            "labels": torch.tensor(item["labels"]),
            "attention_mask": torch.tensor([1] * len(item["input_ids"])),
        }
