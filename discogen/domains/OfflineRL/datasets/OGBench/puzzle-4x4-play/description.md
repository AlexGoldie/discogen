## Puzzle 4x4 Play

**Environment:** `puzzle-4x4-play-singletask-v0`

### Description
An extended sliding puzzle task with a 4x4 grid (16 pieces). This significantly increases the complexity compared to 3x3, requiring longer solution sequences and more sophisticated planning.

### Observation Space
The observation includes:
- Robot state (joint positions, velocities, end-effector pose)
- Positions of all 16 puzzle pieces
- Goal puzzle configuration
- Gripper state

### Action Space
Continuous actions controlling the robot arm and gripper for sliding pieces.

### Reward Structure
- Reward based on number of correctly positioned pieces
- Bonus for completing the puzzle
- Episode terminates on success or timeout

### Dataset
The offline dataset contains 4x4 puzzle manipulation attempts with varying success levels.

### Challenges
- Much larger state space than 3x3 puzzle
- Longer solution sequences required
- Higher combinatorial complexity
- Requires both manipulation skill and puzzle-solving strategy
- Very long-horizon credit assignment
