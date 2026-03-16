DESCRIPTION
JAXUED Minigrid is a JAX-accelerated grid-world maze navigation environment designed for Unsupervised Environment Design research. An agent must navigate through procedurally generated mazes to reach a goal location. The environment provides a partially observable view around the agent, showing only a local window of the maze rather than the full layout. The implementation is fully jittable and runs on GPU or TPU, enabling training on thousands of parallel environments simultaneously. Maze layouts are procedurally generated to create diverse navigation challenges, making the environment ideal for studying curriculum learning, exploration, and generalization in reinforcement learning.

OBSERVATION SPACE
The observation is a 4-dimensional ndarray with shape (view_size, view_size, 3), where view_size is typically 5 or 7. This represents the agent's partial field of view centered on its current position. Each cell in the view contains encoded information about the grid tile:

Channel 0: Tile type ID (wall, empty floor, goal, lava, door, key, etc.)
Channel 1: Color ID for the tile
Channel 2: State information (for example, whether a door is open or closed)

The observations are not image pixels but rather integer encodings of tile properties. The agent can only observe a local window around its current position and orientation, which means it must explore the environment and potentially use memory to navigate successfully through larger mazes.

ACTION SPACE
The action space consists of 6 discrete actions:

0: move_forward - Move one cell in the direction the agent is facing
1: turn_left - Rotate the agent's orientation 90 degrees counterclockwise
2: turn_right - Rotate the agent's orientation 90 degrees clockwise
3: pick_up - Pick up an object located in front of the agent
4: put_down - Drop the currently held object
5: toggle - Activate a switch or open/close a door

In pure maze navigation tasks, typically only actions 0, 1, and 2 are relevant, as the task involves moving to the goal without interacting with objects.

TRANSITION DYNAMICS
The agent operates on a discrete 2D grid, commonly of size 13×13 or 15×15 cells. The agent maintains a directional state, facing one of four cardinal directions: north, south, east, or west. When the move_forward action is taken, the agent attempts to advance one cell in its facing direction, but this movement is blocked if a wall occupies that cell. The turn_left and turn_right actions change the agent's orientation by 90 degrees without changing its position. The goal is reached by moving the agent onto the goal tile. All transitions are fully deterministic given the current state and chosen action. Unlike the Kinetix environments, there is no physics simulation; the environment uses discrete state transitions on the grid.

REWARD
The reward structure depends on the configuration chosen for the environment. In the default sparse reward setting, the agent receives a reward of +1 for reaching the goal tile and 0 at all other timesteps. In the optional dense reward setting, the agent receives a small negative reward per step (for example, -0.01) to encourage efficient solutions, plus +1 for reaching the goal. Some variants use distance-based reward shaping to provide guidance. The sparse reward structure combined with procedural maze generation creates significant exploration challenges, as the agent must discover the goal through exploration without intermediate feedback.

STARTING STATE
The agent is spawned at a designated start position, often near a corner or edge of the maze. The agent's initial orientation is randomly selected from the four cardinal directions. The maze layout is generated procedurally using various algorithms including random obstacle placement, recursive division, depth-first search maze generation, and other methods that can be controlled by Unsupervised Environment Design algorithms. The goal position is placed in the maze, typically positioned far from the start location to create a meaningful challenge. The maze generation process ensures that a valid path exists from the start to the goal.

EPISODE END
The episode ends if either of the following happens:

- Termination: The agent reaches the goal tile
- Truncation: The length of the episode reaches the maximum number of timesteps, which is typically 250 to 500 depending on the maze size and difficulty
