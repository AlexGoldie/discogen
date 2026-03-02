DESCRIPTION
A 3-link arm robot must learn to push a cylinder to a target location. The arm has 7 degrees of freedom and operates in a 2D plane. The goal is to manipulate the cylinder to reach the target position using coordinated arm movements.

OBSERVATION SPACE
The observation is a ndarray with shape (23,) containing:

Joint angles and angular velocities of the 7 arm joints
End-effector position
Cylinder position and velocity
Target position
Distance vectors between relevant objects

ACTION SPACE
The action space consists of 7 continuous actions in the range [-1, 1]:

Torques applied to each of the 7 arm joints

TRANSITION DYNAMICS
2D physics simulation with arm and cylinder interactions
Contact forces between arm and cylinder enable pushing
Cylinder can be pushed around the 2D workspace
Target position remains fixed during episode

REWARD
Large negative reward based on distance from cylinder to target
Small negative reward based on distance from end-effector to cylinder
Small negative reward for control effort
Dense reward signal guides learning

STARTING STATE
Arm starts in a neutral configuration
Cylinder placed at random position in workspace
Target position set randomly in reachable area
Small noise added to initial joint angles

EPISODE END
The episode ends when:

Truncation: The length of the episode reaches max_steps (default: 1000)
(No termination conditions based on task completion)
