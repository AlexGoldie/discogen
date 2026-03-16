## AntMaze Large Navigate

**Environment:** `antmaze-large-navigate-singletask-v0`

### Description
In this task, an 8-DoF quadruped ant robot must navigate through a large maze to reach a goal location. The ant must learn to coordinate its leg movements while planning a path through the maze's corridors and turns.

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
The offline dataset contains trajectories from a mixture of policies:
- Expert navigation demonstrations
- Suboptimal exploration trajectories
- Random actions near goals

### Challenges
- Long-horizon planning through maze corridors
- Sparse rewards make credit assignment difficult
- Distribution shift between offline data and optimal policy
- Requires both low-level locomotion and high-level navigation
