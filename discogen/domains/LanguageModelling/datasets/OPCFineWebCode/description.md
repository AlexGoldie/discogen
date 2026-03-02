DESCRIPTION
OPC-FineWeb-Code is a specialized corpus of code-related web content extracted from FineWeb using fastText-based retrieval in iterative rounds. Originally developed for the OpenCoder pretraining pipeline, this dataset focuses on web pages containing programming tutorials, documentation, code snippets, and technical discussions. The implementation uses the first 20% of the full dataset (~30GB) to balance corpus size with training efficiency.

CONTENT
The dataset consists of code-centric web pages including:

Programming tutorials and documentation
Technical blog posts with code examples
Stack Overflow discussions and solutions
GitHub repository documentation
Code snippets with natural language explanations

DATASET STRUCTURE
Training split: ~80% of the 20% subset (original dataset is ~150GB total)
Validation split: 1% of data (created during preprocessing)
Test split: Remaining ~19% of the 20% subset
Approximately 30GB total size after downsampling
Multiple programming languages and frameworks represented

PREPROCESSING
fastText-based retrieval to identify code-relevant content
Automatic skipping of corrupted or unsafe files
Validation split created from training data (not present in original)
Filtered for code density and technical content quality
