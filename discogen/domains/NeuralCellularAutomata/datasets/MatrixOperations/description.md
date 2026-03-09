# Matrix Operations

This task trains a Neural Cellular Automata to perform matrix operations through local cell interactions. The NCA must learn to compute global matrix operations (transpose, negate, add, multiply) using only local communication between neighboring cells.

## Task Setup

- **Grid size**: 8×8 (each cell corresponds to one matrix element)
- **Input**: Two matrices A and B encoded in state channels 0 and 1 (with elements in [-1,1])
- **Output**: Result matrix read from state channel 2
- **Conditioning**: Operation type encoded one-hot in the last 4 channels (preserved)

## Operations

| Index | Operation | Formula |
|-------|-----------|---------|
| 0 | Transpose | A^T |
| 1 | Negate | -A |
| 2 | Add | A + B |
| 3 | Multiply | A @ B |

## State Encoding

Channel 0: Input matrix A
Channel 1: Input matrix B
Channel 2: Output (NCA writes result here)
Channels 3 to N-5: Hidden state (working memory)
Last 4 channels: Operation one-hot (preserved across steps)

For tasks with this dataset, the matrix size, input and output channels and operations can be accessed in config["matrix"].

## Evaluation

Loss is MSE between predicted output (channel 2) and ground truth result matrix. Lower is better.
