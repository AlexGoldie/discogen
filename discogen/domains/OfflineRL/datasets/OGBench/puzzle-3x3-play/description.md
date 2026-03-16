## Puzzle 3x3 Play

**Environment:** `puzzle-3x3-play-singletask-v0`

### Description
A sliding puzzle manipulation task with a 3x3 grid. The agent must manipulate puzzle pieces to reach a goal configuration, combining physical manipulation skills with puzzle-solving strategy.

### Observation Space
The observation includes:
- Robot state (joint positions, velocities, end-effector pose)
- Positions of all 9 puzzle pieces
- Goal puzzle configuration
- Gripper state

### Action Space
Continuous actions controlling the robot arm and gripper for sliding pieces.

### Reward Structure
- Reward based on number of correctly positioned pieces
- Bonus for completing the puzzle
- Episode terminates on success or timeout

### Dataset
The offline dataset contains puzzle manipulation attempts with varying success levels.

### Challenges
- Requires understanding puzzle structure and valid moves
- Physical manipulation of sliding pieces
- Long-horizon planning for puzzle solution
- Combinatorial complexity of puzzle states
