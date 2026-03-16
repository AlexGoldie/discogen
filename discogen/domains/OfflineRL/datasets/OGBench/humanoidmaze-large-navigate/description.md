## HumanoidMaze Large Navigate

**Environment:** `humanoidmaze-large-navigate-singletask-v0`

### Description
A larger and more challenging version of the humanoid maze navigation task. The 21-DoF humanoid must navigate through an extended maze while maintaining bipedal stability.

### Observation Space
High-dimensional observation including:
- Joint angles and velocities for all 21 joints
- Center of mass position and velocity
- Torso orientation (quaternion)
- Contact forces at feet
- Goal location information

### Action Space
21 continuous actions in [-1, 1] controlling joint torques for the humanoid body.

### Reward Structure
- Sparse reward: +1 when reaching the goal region
- Implicit penalty for falling
- No shaping rewards

### Dataset
The offline dataset contains humanoid navigation trajectories in the large maze environment.

### Challenges
- Extended planning horizon due to larger maze
- High-dimensional control with unstable dynamics
- Long-term credit assignment for sparse rewards
- Balance maintenance throughout extended navigation
