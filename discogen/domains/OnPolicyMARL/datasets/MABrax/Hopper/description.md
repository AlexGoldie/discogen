DESCRIPTION
A 2D one-legged hopper robot must learn to hop forward while maintaining balance. The hopper has 4 degrees of freedom controlling its leg joints and torso. The challenge is to coordinate hopping motions to move forward efficiently without falling over.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls each of the hopper's three joints.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (11,) containing:

Joint angles and angular velocities of the 4 joints
Root position (y-coordinate height and x-coordinate)
Root orientation and angular velocity

Each agent's local observation has shape (9,) after homogenisation. Agents controlling end joints have 8 unique observations; the missing value is zero-padded to match the maximum of 9 (homogenisation_method="max"). Each agent receives a subset of the global observation by selecting specific indices:

Root state (shared, 2 indices): root height and orientation
Neighbor joint states: angle and velocity of adjacent joints in the proximal-to-distal chain
Own joint state (2 indices): angle and velocity of the controlled joint
Root body velocities (shared, 3 indices): global body velocities
Own joint supplementary (1 index): an additional joint-specific value

All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The action space consists of 3 continuous actions in the range [-1, 1]:

Torques applied to the thigh, leg, and foot joints

Each agent controls 1 action (joint torque) for its assigned joint (in proximal-to-distal order):

agent_0 → thigh_joint (joint 0)
agent_1 → leg_joint (joint 1)
agent_2 → foot_joint (joint 2)

TRANSITION DYNAMICS
2D physics simulation in the sagittal plane
Single point of ground contact through the foot
Must maintain balance to avoid falling
Forward progress through hopping motions

REWARD
Positive reward for forward velocity
Small negative reward for energy expenditure (control cost)
Reward for staying alive (not falling)
Episode reward typically ranges from 0 to 3500+
All agents receive the same joint reward.

STARTING STATE
Hopper starts in an upright standing position
Small random noise added to initial joint angles and velocities
Root height starts above ground level

EPISODE END
The episode ends if either of the following happens:

Termination: Hopper falls over (height below threshold or extreme angle)
Truncation: The length of the episode reaches max_steps (default: 1000)
