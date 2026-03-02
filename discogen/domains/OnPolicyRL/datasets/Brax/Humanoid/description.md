DESCRIPTION
A 3D humanoid robot with 21 degrees of freedom must learn to walk forward while maintaining balance and upright posture. The humanoid has a complex structure with torso, arms, and legs. The goal is to achieve stable bipedal locomotion.

OBSERVATION SPACE
The observation is a ndarray with shape (376,) containing:

Joint angles and angular velocities of all 21 joints
Root position and orientation (3D position and quaternion)
Root linear and angular velocities
Center of mass information

ACTION SPACE
The action space consists of 17 continuous actions in the range [-1, 1]:

Torques applied to all joints including spine, arms, hips, knees, and ankles

TRANSITION DYNAMICS
Full 3D physics simulation with complex multi-body dynamics
Multiple contact points with ground (feet, potentially other body parts)
Must maintain balance in 3D space
Coordinated movement of all limbs required

REWARD
Positive reward for forward velocity
Reward for maintaining upright posture
Small negative reward for energy expenditure (control cost)
Penalty for impact forces (falling)
Episode reward typically ranges from 0 to 5000+

STARTING STATE
Humanoid starts in upright standing position
Small random noise added to all joint positions and velocities
Center of mass positioned above support polygon

EPISODE END
The episode ends if either of the following happens:

Termination: Humanoid falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
