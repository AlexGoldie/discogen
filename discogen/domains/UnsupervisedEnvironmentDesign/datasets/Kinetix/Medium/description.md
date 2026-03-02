DESCRIPTION
Kinetix-Medium is a 2D physics-based reinforcement learning environment that supports more complex tasks than the Small variant. Agents control physical systems using motors, which apply torque, and thrusters, which apply force, to make green objects touch blue targets while avoiding red hazards. Medium environments can represent multi-stage manipulation tasks, navigation through obstacle courses, and coordination of articulated mechanisms with more objects and actuators than Small. The environment is powered by the Jax2D physics engine and runs on GPU or TPU via JAX for efficient parallel simulation.

OBSERVATION SPACE
The observation is an ndarray containing structured state information:

- Positions (x, y coordinates) of all rigid bodies
- Linear velocities for all bodies
- Angular positions (orientations) for all bodies
- Angular velocities for all bodies
- Complete motor state information (angles, velocities, applied torques)
- Complete thruster state information (positions, applied forces)
- Object type encodings
- Collision and contact force information
- Distance vectors between important object pairs

Medium environments support more objects than Small, with typical observation dimensionality ranging from 150 to 300 dimensions.

ACTION SPACE
The action space provides control over motors and thrusters:

- Continuous: Torque values in [-1, 1] for motors, force values in [0, 1] for thrusters
- Multi-discrete: Each actuator selects from multiple discrete control levels

Medium environments typically have 3 to 6 total actuators, allowing for more sophisticated multi-component control strategies.

TRANSITION DYNAMICS
The environment uses the Jax2D engine for full 2D rigid body physics simulation. The simulation handles interactions between larger numbers of bodies than Small environments. Collision detection operates across all object pairs, with impulse-based collision resolution incorporating friction and restitution properties. Motor torques create rotational forces at hinged joints between connected bodies. Thruster forces are applied as linear impulses to bodies in specified directions. Complex emergent behaviors can arise from multi-body interactions. Despite the increased complexity, the simulation remains hardware-accelerated via JAX for efficient parallel execution.

REWARD
The agent receives a reward of +1 upon successful task completion when the green object touches the blue object. At all other timesteps, the reward is 0. There are no negative rewards or step penalties. The episode terminates immediately upon success or failure.

STARTING STATE
The green controllable object or objects are positioned at the starting configuration. The blue goal object is placed at the target location. Red hazard objects are positioned as obstacles or failure conditions that must be avoided. Gray neutral objects form structures, walls, or movable elements in the environment. The configuration may include articulated structures with multiple connected bodies. Initial velocities are typically zero unless the level specifies otherwise. Procedural generation can create varied starting configurations across different level instances.

EPISODE END
The episode ends if any of the following happens:

- Termination (success): The green object makes contact with the blue goal object
- Termination (failure): The green object makes contact with a red hazard object
- Truncation: The length of the episode reaches the maximum number of timesteps (default: 1000)
