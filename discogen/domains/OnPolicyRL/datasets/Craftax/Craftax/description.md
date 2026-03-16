DESCRIPTION
Craftax is an RL environment written entirely in JAX. Craftax reimplements and significantly extends the game mechanics of Crafter, taking inspiration from roguelike games such as NetHack. Craftax is an extended version that builds upon Crafter's foundation with additional mechanics inspired by NetHack. The environment features more complex gameplay including multiple dungeon levels, enhanced combat systems, magic spells, and expanded crafting trees compared to Craftax-Classic.

OBSERVATION SPACE
The observation includes multiple components representing the enhanced game state:
- Visual observation of the surrounding world (64x64 RGB image by default)
- Extended inventory system with more item types
- Character statistics including health, mana, experience
- Equipment and spell information
- Achievement and quest progress indicators
- Dungeon level and exploration status

ACTION SPACE
The action space consists of expanded discrete actions including:

Num Action
0 no-op
1-4 movement (left, right, up, down)
5 interact/use
6 sleep
7 cast spell
8-40+ various crafting, combat, and magical actions

TRANSITION DYNAMICS
- Enhanced movement and interaction system
- Multi-level dungeon exploration with stairs
- Complex combat with multiple enemy types and abilities
- Magic system with spells and mana management
- Extended crafting with enchantments and upgrades
- Experience and leveling system
- Enhanced day/night and weather effects

REWARD
- Achievement-based reward system with expanded objectives
- 62 different achievements available across combat, exploration, crafting, and magic
- Maximum possible reward of 226 points
- Hierarchical achievement structure with dependencies

STARTING STATE
- Player spawns on surface level of procedurally generated world
- Empty inventory and basic starting equipment
- Level 1 character with minimal stats
- Full health and mana at start

EPISODE END
The episode ends if either of the following happens:
1. Termination: Player health reaches zero
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000000)

STATE SPACE
Both environments maintain complex internal state including:
- World map with terrain, resources, and entities
- Player position, stats, and inventory
- Enemy positions and states
- Time of day and weather conditions
- Achievement completion status
- Procedural generation seeds and world history
