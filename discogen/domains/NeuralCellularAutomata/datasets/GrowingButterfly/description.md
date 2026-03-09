# Growing Butterfly

Grow a butterfly emoji from a single seed cell. The NCA learns local update rules that, when applied iteratively, produce the target pattern through emergent morphogenesis.

## Task Setup

- **Grid size**: 72×72 (40×40 target with 16px padding on each side)
- **Initial state**: Single alive cell in the center of an empty grid
- **Target**: Butterfly emoji (🦋) rendered as RGBA image

## State Encoding

- **Channels 0-11**: Hidden state (working memory, not supervised)
- **Channels 12-14**: RGB color values
- **Channel 15**: Alpha/alive channel

The last 4 channels (RGBA) are compared against the target. Pixel values are in [0, 1].

## Initialization

All channels start at zero except the center cell, which has its alive channel (channel 15) set to 1.0.

## Evaluation

MSE between NCA output (last 4 channels) and target RGBA image. Lower is better.
