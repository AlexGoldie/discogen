DESCRIPTION
Craftax-Classic is a ground-up rewrite of Crafter in JAX that runs up to 250x faster than the Python-native original. It offers significant performance improvements while maintaining the core gameplay. The player spawns in a 2D world and must survive by gathering resources, crafting tools, building shelter, and defending against enemies. The environment features procedurally generated worlds with various biomes, creatures, and resources.

OBSERVATION SPACE
The observation includes multiple components representing the game state:
- Visual observation of the surrounding world (64x64 RGB image by default)
- Inventory information showing collected items and tools
- Health and other vital statistics
- Achievement progress indicators

ACTION SPACE
The action space consists of discrete actions including:

Num Action
0 no-op
1 move left
2 move right
3 move up
4 move down
5 do (interact/use)
6 sleep
7-23 various crafting and placement actions

TRANSITION DYNAMICS
- Player can move in four cardinal directions
- Resources can be collected by walking over them or using tools
- Crafting requires specific combinations of materials
- Health decreases over time and from enemy attacks
- Day/night cycle affects visibility and enemy behavior
- Achievements unlock based on completed tasks

REWARD
- Sparse rewards based on achievement unlocking
- 22 different achievements available (collect wood, make tools, defeat monsters, etc.)
- Maximum possible reward of 22 points
- No step-based penalties

STARTING STATE
- Player spawns in a random location in the procedurally generated world
- Initial inventory is empty
- Full health at start
- Daytime at episode beginning

EPISODE END
The episode ends if either of the following happens:
1. Termination: Player health reaches zero
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000000)
