DESCRIPTION
In Brax Ant, a quadrupedal ant robot with 8 degrees of freedom must learn to walk forward as quickly as possible. The ant consists of a torso with four legs, each having two joints (hip and ankle). The goal is to coordinate the leg movements to achieve stable and fast forward locomotion while maintaining balance.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: one agent controls each of the ant's four legs.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (87,) containing:

Joint angles and angular velocities of the 8 joints
Position and orientation of the torso (x, y, z position and quaternion orientation)
Linear and angular velocities of the torso
Contact forces at the feet

Each agent's local observation has shape (18,), uniform across all 4 agents. Each agent receives a subset of the global observation by selecting specific indices:

Root/body state (shared, 15 indices): root body position, quaternion orientation, linear and angular velocities, plus angle and velocity of the 3 neighboring leg joints
Own joint state (3 indices): hip joint angle+velocity and ankle joint angle+velocity for the controlled leg, plus 1 foot contact observation

All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space consists of 8 continuous actions in the range [-1, 1]:

4 hip joint torques (one for each leg)
4 ankle joint torques (one for each leg)

Each agent controls 2 actions (hip torque, ankle torque) for the joints of its assigned leg:

agent_0 → hip joint 0 + ankle joint 1 (leg 0)
agent_1 → hip joint 2 + ankle joint 3 (leg 1)
agent_2 → hip joint 4 + ankle joint 5 (leg 2)
agent_3 → hip joint 6 + ankle joint 7 (leg 3)

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
All agents receive the same joint reward.

STARTING STATE

Ant starts in an upright position at the origin
Small random noise is added to initial joint positions and velocities
All joints begin near their neutral positions

EPISODE END
The episode ends if either of the following happens:

Termination: Ant falls over (torso height below threshold or extreme orientation)
Truncation: The length of the episode reaches max_steps (default: 1000)
