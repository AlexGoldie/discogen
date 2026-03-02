DESCRIPTION
A 3D humanoid robot with 21 degrees of freedom must learn to walk forward while maintaining balance and upright posture. The humanoid has a complex structure with torso, arms, and legs. The goal is to achieve stable bipedal locomotion.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls the upper body and one agent controls the lower body.

OBSERVATION SPACE
The global observation is a ndarray with shape (376,) containing:

Joint angles and angular velocities of all 21 joints
Root position and orientation (3D position and quaternion)
Root linear and angular velocities
Center of mass information

Each agent receives a subset of the 376 global observation values by selecting specific indices:

agent_0 (upper body): local observation has shape (248,), selecting 248 of the 376 global indices
agent_1 (lower body): local observation has shape (176,), selecting 176 of the 376 global indices

With homogenisation_method="max", both agents receive shape-(248,) observations; agent_1's observation is zero-padded from 176 to 248. Each agent's local observation covers the root/body state (shared), its own controlled joint states, and neighboring joint states. All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space of all agents consists of 17 continuous actions in the range [-0.4, 0.4]:

Torques applied to all joints including spine, arms, hips, knees, and ankles

The 17 actions are split non-uniformly between agents based on body region:

agent_0 (upper body): controls 9 joints (spine + arms), corresponding to global action indices [0, 1, 2, 11, 12, 13, 14, 15, 16]
agent_1 (lower body): controls 8 joints (hips + legs), corresponding to global action indices [3, 4, 5, 6, 7, 8, 9, 10]

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
All agents receive the same joint reward.

EPISODE END
The episode ends if either of the following happens:

Termination: Humanoid falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
