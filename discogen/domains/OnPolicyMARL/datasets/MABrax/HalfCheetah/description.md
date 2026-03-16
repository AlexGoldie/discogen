DESCRIPTION
In Brax HalfCheetah, a 2D cheetah robot constrained to move in the sagittal plane must learn to run forward as quickly as possible. The cheetah has 6 degrees of freedom controlling its spine and legs. The goal is to achieve maximum forward running speed while maintaining stability.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls each of the cheetah's six joints.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (18,) containing:

Joint angles and angular velocities of the 6 joints
Root position (x-coordinate only, no y or z)
Root orientation and angular velocity
Velocity information

Each agent's local observation has shape (9,) after homogenisation. Agents controlling end/isolated joints have 8 unique observations; the missing value is zero-padded to match the maximum of 9 (homogenisation_method="max"). Each agent receives a subset of the global observation by selecting specific indices:

Root state (shared, 2 indices): root x-velocity and angular velocity
Neighbor joint states (2–3 indices): angle and velocity of adjacent joints in the kinematic chain
Own joint state (2 indices): angle and velocity of the controlled joint
Root body extra (3 indices): additional shared body state values

All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space consists of 6 continuous actions in the range [-1, 1]:

Torques applied to each of the 6 joints (spine, thigh, shin, foot joints)

Each agent controls 1 action (joint torque) for its assigned joint (in kinematic order, back to front):

agent_0 → bthigh (back thigh, joint 0)
agent_1 → bshin (back shin, joint 1)
agent_2 → bfoot (back foot, joint 2)
agent_3 → fthigh (front thigh, joint 3)
agent_4 → fshin (front shin, joint 4)
agent_5 → ffoot (front foot, joint 5)

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
All agents receive the same joint reward.

STARTING STATE

Cheetah starts in a neutral standing position
Small random noise added to initial joint angles and velocities
Root position starts at origin

EPISODE END
The episode ends when:

Truncation: The length of the episode reaches max_steps (default: 1000)
(No termination conditions - cheetah cannot fall in 2D)
