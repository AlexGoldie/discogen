## HumanoidMaze Medium Navigate

**Environment:** `humanoidmaze-medium-navigate-singletask-v0`

### Description
A humanoid robot with 21 degrees of freedom must navigate through a medium-sized maze. This task combines the challenge of bipedal locomotion with maze navigation, requiring the agent to maintain balance while planning paths.

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
- Implicit penalty for falling (episode termination)
- No shaping rewards

### Dataset
The offline dataset contains trajectories from humanoid policies with varying skill levels navigating the maze.

### Challenges
- High-dimensional action space (21 DoF)
- Unstable dynamics requiring balance maintenance
- Combines locomotion learning with navigation
- Sparse rewards over complex movement sequences
