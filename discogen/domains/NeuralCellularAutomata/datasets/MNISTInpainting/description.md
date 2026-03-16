# MNIST Inpainting

Reconstruct partially masked MNIST digits. Given an image with 50% of pixels randomly masked, the NCA must infer the missing pixels by propagating information from visible regions.

## Task Setup

- **Grid size**: 28×28 (matches MNIST image dimensions)
- **Initial state**: MNIST digit with 50% of pixels randomly set to zero
- **Target**: Original unmasked MNIST image

## State Encoding

- **Channels 0-30**: Hidden state (working memory)
- **Channel 31**: Grayscale pixel value (starts as masked input, updated by NCA)

All channels can be modified by the NCA (`preserve_channels: 0`).

## Initialization

The masked MNIST image is placed in the last channel. Mask is applied randomly with 50% of pixels zeroed. All other channels start at zero.

## Evaluation

MSE between NCA output (last channel) and original unmasked image. Lower is better.
