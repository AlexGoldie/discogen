## AntMaze Giant Navigate

**Environment:** `antmaze-giant-navigate-singletask-v0`

### Description
A significantly larger and more complex version of the AntMaze task. The 8-DoF ant robot must navigate through a giant maze with longer corridors and more complex path planning requirements.

### Observation Space
The observation includes:
- Joint angles and angular velocities (8 joints × 2 = 16 dims)
- Torso position and orientation
- Linear and angular velocities
- Goal location (relative to agent)
- Maze structure information

### Action Space
8 continuous actions in [-1, 1] controlling torques for:
- 4 hip joints (one per leg)
- 4 ankle joints (one per leg)

### Reward Structure
- Sparse reward: +1 when reaching the goal region
- No intermediate shaping rewards
- Episode terminates on goal reach or timeout

### Dataset
The offline dataset contains trajectories from a mixture of policies navigating the giant maze.

### Challenges
- Very long-horizon planning
- Sparse rewards over extended trajectories
- Large state space due to maze size
- Requires robust long-term credit assignment
