## Cube Double Play

**Environment:** `cube-double-play-singletask-v0`

### Description
An extended manipulation task involving two cubes. The agent must manipulate both cubes to achieve goal configurations, requiring sequential or coordinated manipulation strategies.

### Observation Space
The observation includes:
- Robot end-effector position and orientation
- Joint positions and velocities
- Positions and orientations of both cubes
- Goal positions for both cubes
- Gripper state

### Action Space
Continuous actions controlling the robot arm and gripper.

### Reward Structure
- Reward based on both cubes' proximity to goals
- Partial credit for individual cube placement
- Episode terminates on full success or timeout

### Dataset
The offline dataset contains trajectories of manipulating two cubes, with various strategies for handling multiple objects.

### Challenges
- Requires planning for multiple objects
- Order of manipulation may matter
- More complex state space than single cube
- Longer horizon for complete task
