DESCRIPTION
In Brax Walker2d, a 2D bipedal walker robot must learn to walk forward while maintaining balance. The walker has 6 degrees of freedom controlling its legs, thighs, and torso. The goal is to achieve stable forward locomotion without falling over.

OBSERVATION SPACE
The observation is a ndarray with shape (17,) containing:

Joint angles and angular velocities of the 6 joints
Root height and orientation
Root linear and angular velocities
Contact forces with the ground

ACTION SPACE
The action space consists of 6 continuous actions in the range [-1, 1]:

Torques applied to thigh, leg, and foot joints for both legs

TRANSITION DYNAMICS

2D physics simulation in the sagittal plane
Bipedal contact with ground through both feet
Must maintain balance to avoid falling
Forward locomotion through coordinated leg movements

REWARD

Positive reward for forward velocity
Small negative reward for energy expenditure (control cost)
Reward for staying alive (not falling)
Episode reward typically ranges from 0 to 5000+

STARTING STATE

Walker starts in upright standing position
Small random noise added to initial joint angles and velocities
Both feet in contact with ground

EPISODE END
The episode ends if either of the following happens:

Termination: Walker falls over (torso height below threshold or extreme angle)
Truncation: The length of the episode reaches max_steps (default: 1000)
