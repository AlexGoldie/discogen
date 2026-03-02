Multi Particle Environments (MPE) are a set of communication oriented environment where particle agents can (sometimes) move, communicate, see each other, push each other around, and interact with fixed landmarks.

**Observation Space**
The exact observation varies for each environment, but in general it is a vector of agent/landmark positions and velocities along with any communication values.

Agent observations: [self_vel, self_pos, landmark_rel_positions, other_agent_rel_positions, communication]
Observation shape: (18,)

**Action Space**
The environments use discrete actions.

Represents the combination of movement and communication actions. Agents that can move select a value 0-4 corresponding to [do nothing, down, up, left, right], while agents that can communicate choose between a number of communication options. The agents' abilities and with the number of communication options varies with the envrionments.

Agent action space: [no_action, move_left, move_right, move_down, move_up]
Action shape: (5,)

**Spread**
This environment has N agents, N landmarks (default N=3). At a high level, agents must learn to cover all the landmarks while avoiding collisions.

More specifically, all agents are globally rewarded based on how far the closest agent is to each landmark (sum of the minimum distances). Locally, the agents are penalized if they collide with other agents (-1 for each collision). The relative weights of these rewards can be controlled with the local_ratio parameter.
