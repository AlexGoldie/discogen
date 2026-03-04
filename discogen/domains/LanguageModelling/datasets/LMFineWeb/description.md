DESCRIPTION
LMFineWeb (10B variant) is a curated web text corpus derived from the larger FineWeb dataset by Hugging Face. The dataset prioritizes quality through aggressive filtering of Common Crawl data, removing low-quality content, duplicates, and non-English text. This 10B token variant represents a substantial downsample of the full dataset, designed specifically for rapid experimentation and iteration during language model development.

CONTENT
The dataset consists of cleaned web pages spanning diverse topics and domains, with each example containing:

High-quality natural language text
Metadata including source URLs and filtering scores
Multi-sentence coherent passages suitable for language modeling

DATASET STRUCTURE
Training split: Approximately 10 billion tokens
Downsampled from the full FineWeb corpus for efficiency
Preserves the quality distribution of the parent dataset
Suitable for pretraining small to medium-sized language models

PREPROCESSING
Extensive deduplication at document and paragraph levels
Language filtering to retain primarily English content
Quality filtering based on heuristics and model-based scoring
Removal of adult content, spam, and machine-generated text
