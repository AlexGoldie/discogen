DESCRIPTION
A 2-link arm robot must learn to reach a target location with its end-effector. The arm operates in a 2D plane and has 2 degrees of freedom. The goal is to position the end-effector as close as possible to a randomly placed target.

OBSERVATION SPACE
The observation is a ndarray with shape (11,) containing:

Joint angles and angular velocities of the 2 joints
End-effector position (x, y coordinates)
Target position (x, y coordinates)
Distance vector from end-effector to target

ACTION SPACE
The action space consists of 2 continuous actions in the range [-1, 1]:

Torques applied to the shoulder and elbow joints

TRANSITION DYNAMICS
2D kinematic chain with two revolute joints
End-effector position determined by forward kinematics
No contact forces or external objects
Simple point-to-point reaching task

REWARD
Large negative reward proportional to distance from end-effector to target
Small negative reward for control effort (action magnitude)
Dense reward signal provides continuous feedback

STARTING STATE
Arm starts in random initial configuration
Target placed randomly within reachable workspace
Joint angles initialized with small random noise

EPISODE END
The episode ends when:

Truncation: The length of the episode reaches max_steps (default: 50)
(No termination conditions - short episodes for reaching task)
