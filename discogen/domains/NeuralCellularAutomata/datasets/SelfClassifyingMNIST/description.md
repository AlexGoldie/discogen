# Self-Classifying MNIST

Classify MNIST digits using only local communication. Each cell must output classification logits for the digit class, reaching global consensus through local message passing.

## Task Setup

- **Grid size**: 28×28 (matches MNIST image dimensions)
- **Initial state**: MNIST digit in last channel, all other channels zero
- **Target**: Per-pixel one-hot encoding of the digit class

## State Encoding

- **Channels 0-9**: Classification logits (one per digit class)
- **Channels 10-18**: Hidden state (working memory)
- **Channel 19**: Input MNIST pixel value (preserved across updates)

For pixels where the digit is present (value >= 0.1), the target is the one-hot vector for the true class. For background pixels, the target is all zeros.

## Configuration

- `preserve_channels: 1` (input image preserved in last channel)
- `use_alive_masking: False`
- `channel_size: 20`

## Evaluation

Cross-entropy loss between the classification logits (channels 0-9) and the one-hot target. Lower is better.
