## Cube Single Play

**Environment:** `cube-single-play-singletask-v0`

### Description
A robotic manipulation task where an agent must manipulate a single cube to achieve goal configurations. The task involves precise control for grasping, lifting, and placing the cube.

### Observation Space
The observation includes:
- Robot end-effector position and orientation
- Joint positions and velocities
- Cube position and orientation
- Goal cube position/orientation
- Gripper state

### Action Space
Continuous actions controlling the robot arm and gripper.

### Reward Structure
- Reward based on cube proximity to goal
- Bonus for successful placement
- Episode terminates on success or timeout

### Dataset
The offline dataset contains cube manipulation trajectories from play data, including both successful and exploratory behaviors.

### Challenges
- Precise manipulation requires fine motor control
- Contact dynamics add complexity
- Multiple valid strategies for achieving goals
- Requires learning from mixed-quality demonstrations
