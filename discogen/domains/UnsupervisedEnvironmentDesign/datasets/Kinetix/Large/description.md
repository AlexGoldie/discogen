DESCRIPTION
Kinetix-Large is the most complex variant of the 2D physics-based reinforcement learning environment, supporting sophisticated tasks with many objects and actuators. The objective remains consistent across all Kinetix variants: control motors and thrusters to make green objects touch blue targets without touching red hazards. Large environments can represent complex vehicle control, intricate manipulation chains, multi-stage puzzles, and tasks requiring extended sequences of precise actions. These environments push the boundaries of physical reasoning and long-horizon planning while maintaining computational efficiency through the JAX-accelerated Jax2D physics engine.

OBSERVATION SPACE
The observation is an ndarray containing comprehensive physical state information:

- Positions (x, y coordinates) for all rigid bodies in the scene
- Linear velocities for all bodies
- Angular positions (orientations) for all bodies
- Angular velocities for all bodies
- Full articulation state for all joints and connections
- Complete motor states (positions, velocities, torques)
- Complete thruster states (positions, forces, orientations)
- Object type identifiers for all bodies
- Detailed contact and collision force information
- Potentially includes geometric features or semantic encodings

Large environments have the highest observation dimensionality, typically ranging from 300 to 600 or more dimensions depending on the number of objects and task complexity.

ACTION SPACE
The action space controls a large number of actuators:

- Continuous: Independent control of 6 to 12 or more motors with torques in [-1, 1] and thrusters with forces in [0, 1]
- Multi-discrete: Each of the many actuators selects from discrete control options

Large environments typically have 6 to 12 or more total actuators, requiring sophisticated coordination strategies to achieve task objectives.

TRANSITION DYNAMICS
The environment uses the full-featured Jax2D engine for 2D rigid body physics simulation. The simulation handles complex multi-body systems including chains, hinges, wheels, and compound structures. Collision detection and resolution operate across many simultaneous contacts between objects. Motor torques drive rotation at revolute joints connecting bodies. Thruster forces provide linear impulses to bodies in specified directions. The simulation incorporates realistic material properties such as friction, restitution, and density. Emergent complex behaviors arise from interactions among many components. Despite the computational complexity, the simulation maintains efficiency through JAX parallelization on GPU or TPU hardware.

REWARD
The agent receives a reward of +1 when the goal is achieved by having the green object touch the blue object. At all other timesteps, the reward is 0. There are no intermediate rewards or step penalties. Success often requires executing long sequences of coordinated actions across multiple actuators.

STARTING STATE
The environment begins with complex initial configurations containing many objects. Green controllable objects or systems are positioned at their starting locations. The blue goal object is placed at the target location. Red hazard objects create challenging obstacle layouts that must be navigated or avoided. Gray objects form elaborate structures such as mazes, bridges, or mechanical contraptions. The initial state may include vehicles, articulated arms, or other multi-component systems. Levels can feature procedurally generated layouts with diverse topologies and challenges. Initial velocities are usually zero unless the level specifies otherwise.

EPISODE END
The episode ends if any of the following happens:

- Termination (success): The green object makes contact with the blue goal object
- Termination (failure): The green object makes contact with a red hazard object
- Truncation: The length of the episode reaches the maximum number of timesteps (default: 1000)
