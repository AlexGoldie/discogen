DESCRIPTION
In Brax Walker2d, a 2D bipedal walker robot must learn to walk forward while maintaining balance. The walker has 6 degrees of freedom controlling its legs, thighs, and torso. The goal is to achieve stable forward locomotion without falling over.

MULTI AGENT ENVIRONMENT
This environment is multi-agent: each agent controls one of the walker's two legs.

OBSERVATION SPACE
The combined observation of all agents is a ndarray with shape (17,) containing:

Joint angles and angular velocities of the 6 joints
Root height and orientation
Root linear and angular velocities
Contact forces with the ground

Each agent's local observation has shape (10,), equal for both agents. Each agent receives a subset of the global observation by selecting specific indices:

Root height (shared, 1 index): the walker's torso height
Own and neighbor joint states (7 indices): a subset of the 6 joint angles and velocities covering the agent's 3 controlled joints plus adjacent joints
Root velocities and contact (shared, 5 indices): root linear/angular velocities and foot contact information

All observations are continuous numbers in the range [-inf, inf].

ACTION SPACE
The combined action space of all agents consists of 6 continuous actions in the range [-1, 1]:

Torques applied to thigh, leg, and foot joints for both legs

Each agent controls 3 actions (thigh torque, leg torque, foot torque) for the joints of its assigned leg:

agent_0 → right_thigh (joint 0), right_leg (joint 1), right_foot (joint 2)
agent_1 → left_thigh (joint 3), left_leg (joint 4), left_foot (joint 5)

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
All agents receive the same joint reward.

STARTING STATE

Walker starts in upright standing position
Small random noise added to initial joint angles and velocities
Both feet in contact with ground

EPISODE END
The episode ends if either of the following happens:

Termination: Walker falls over (torso height below threshold or extreme angle)
Truncation: The length of the episode reaches max_steps (default: 1000)
