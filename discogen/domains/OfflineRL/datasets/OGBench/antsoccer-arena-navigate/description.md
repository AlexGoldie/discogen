## AntSoccer Arena Navigate

**Environment:** `antsoccer-arena-navigate-singletask-v0`

### Description
An ant robot in a soccer arena environment must navigate to goal positions. The arena provides an open space with boundaries, requiring the agent to learn efficient locomotion and goal-directed navigation.

### Observation Space
The observation includes:
- Joint angles and angular velocities (8 joints)
- Torso position and orientation
- Ball position (if applicable)
- Goal location information
- Arena boundary information

### Action Space
8 continuous actions in [-1, 1] controlling the ant's joint torques.

### Reward Structure
- Sparse reward for reaching goal regions
- Episode terminates on success or timeout

### Dataset
The offline dataset contains ant navigation trajectories in the soccer arena setting.

### Challenges
- Open arena requires efficient path planning
- Multiple potential goal locations
- Long-horizon navigation with sparse rewards
- Requires generalization across goal positions
