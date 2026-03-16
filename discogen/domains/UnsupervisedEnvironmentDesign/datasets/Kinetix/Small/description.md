DESCRIPTION
Kinetix-Small is a 2D physics-based reinforcement learning environment where agents learn to control physical objects through motors and thrusters. The universal objective is to make a green controllable object touch a blue target object without the green object touching any red hazard objects. This Small variant represents simpler tasks with fewer objects and limited actuators, making it suitable for learning basic physical reasoning and control. The environment uses the Jax2D physics engine and runs entirely on GPU or TPU via JAX, enabling fast parallel simulation of thousands of environments simultaneously.

OBSERVATION SPACE
The observation is an ndarray containing structured state information representing the physical world:
- Positions (x, y coordinates) of all rigid bodies in the scene
- Linear velocities of all bodies
- Angular positions (orientations) of all bodies
- Angular velocities of all bodies
- Motor states including current angles and angular velocities
- Thruster states
- Object type identifiers (green=controllable, blue=goal, red=hazard, gray=neutral)
- Contact information between colliding objects

The exact dimensionality depends on the specific level but typically ranges from 50 to 150 dimensions for Small environments.

ACTION SPACE
The action space can be either continuous or multi-discrete depending on configuration:

- Continuous: Each motor receives a torque value in [-1, 1], each thruster receives a force value in [0, 1]
- Multi-discrete: Each actuator selects from several discrete torque or force levels

Small environments typically have 1 to 3 actuators in total, combining motors and thrusters.

TRANSITION DYNAMICS
The physics simulation advances using Jax2D, a 2D rigid body physics engine built specifically for Kinetix. The engine uses discrete Euler integration to update velocities and positions at each timestep. When objects collide, the engine performs collision detection and applies impulse-based collision resolution to handle contacts realistically. Motors apply rotational torque at joints connecting bodies, while thrusters apply directional force to bodies. The simulation includes realistic friction and restitution coefficients to model material properties. All computation runs on accelerators through JAX, allowing massive parallelization across environments.

REWARD
The agent receives a reward of +1 when the green controllable object makes contact with the blue goal object. At all other timesteps, the reward is 0. There are no negative rewards or step penalties.

STARTING STATE
The green controllable object is placed at a designated starting position, while the blue goal object is positioned at the target location. Red hazard objects, if present in the level, are placed as obstacles that must be avoided. Gray neutral objects and walls provide environmental structure such as barriers or platforms. Some levels include small random perturbations to initial positions to increase diversity. All objects typically start at rest with zero velocity unless the level specifies otherwise.

EPISODE END
The episode ends if any of the following happens:

- Termination (success): The green object touches the blue goal object
- Termination (failure): The green object touches a red hazard object
- Truncation: The length of the episode reaches the maximum number of timesteps (default: 1000)
