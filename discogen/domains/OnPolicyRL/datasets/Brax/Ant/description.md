DESCRIPTION
In Brax Ant, a quadrupedal ant robot with 8 degrees of freedom must learn to walk forward as quickly as possible. The ant consists of a torso with four legs, each having two joints (hip and ankle). The goal is to coordinate the leg movements to achieve stable and fast forward locomotion while maintaining balance.

OBSERVATION SPACE
The observation is a ndarray with shape (27,) containing:

Joint angles and angular velocities of the 8 joints
Position and orientation of the torso (x, y, z position and quaternion orientation)
Linear and angular velocities of the torso

ACTION SPACE
The action space consists of 8 continuous actions in the range [-1, 1]:

4 hip joint torques (one for each leg)
4 ankle joint torques (one for each leg)

TRANSITION DYNAMICS

Physics simulation advances the ant's state based on applied torques
Contact forces are computed when feet touch the ground
The ant must maintain balance to avoid falling
Forward velocity is encouraged through the reward function

REWARD

Positive reward for forward velocity (x-direction)
Small negative reward for energy expenditure (control cost)
Negative reward for deviation from upright posture
Episode reward typically ranges from 0 to 6000+

STARTING STATE

Ant starts in an upright position at the origin
Small random noise is added to initial joint positions and velocities
All joints begin near their neutral positions

EPISODE END
The episode ends if either of the following happens:

Termination: Ant falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
