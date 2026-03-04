DESCRIPTION
In Brax HalfCheetah, a 2D cheetah robot constrained to move in the sagittal plane must learn to run forward as quickly as possible. The cheetah has 6 degrees of freedom controlling its spine and legs. The goal is to achieve maximum forward running speed while maintaining stability.

OBSERVATION SPACE
The observation is a ndarray with shape (18,) containing:

Joint angles and angular velocities of the 6 joints
Root position (x-coordinate only, no y or z)
Root orientation and angular velocity
Velocity information

ACTION SPACE
The action space consists of 6 continuous actions in the range [-1, 1]:

Torques applied to each of the 6 joints (spine, thigh, shin, foot joints)

TRANSITION DYNAMICS

2D physics simulation in the sagittal plane
Cheetah cannot fall over due to 2D constraint
Forward locomotion achieved through coordinated joint movements
Ground contact forces affect the motion

REWARD

Large positive reward for forward velocity
Small negative reward for energy expenditure (control cost)
No termination penalty (cheetah cannot fall)
Episode reward typically ranges from 0 to 12000+

STARTING STATE

Cheetah starts in a neutral standing position
Small random noise added to initial joint angles and velocities
Root position starts at origin

EPISODE END
The episode ends when:

Truncation: The length of the episode reaches max_steps (default: 1000)
(No termination conditions - cheetah cannot fall in 2D)
