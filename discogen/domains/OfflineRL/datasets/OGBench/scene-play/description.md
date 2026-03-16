## Scene Play

**Environment:** `scene-play-singletask-v0`

### Description
A complex manipulation task involving multiple objects in a scene. The agent must arrange objects to match a goal scene configuration, requiring understanding of spatial relationships and sequential manipulation.

### Observation Space
The observation includes:
- Robot state (joint positions, velocities, end-effector pose)
- Positions and orientations of all scene objects
- Goal scene configuration
- Gripper state

### Action Space
Continuous actions controlling the robot arm and gripper.

### Reward Structure
- Reward based on scene similarity to goal
- Partial credit for correctly placed objects
- Episode terminates on success or timeout

### Dataset
The offline dataset contains scene manipulation trajectories with various object arrangements.

### Challenges
- Multiple objects to manage simultaneously
- Requires understanding of goal scene structure
- Order of operations can affect feasibility
- Complex planning over object interactions
