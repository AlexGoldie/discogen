DESCRIPTION
Asterix MinAtar is a simplified version where the player moves freely along 4 cardinal directions. Enemies and treasure spawn from the sides. A reward of +1 is given for picking up treasure. The goal is to guide Asterix to avoid enemies and collect as many treasure objects as possible.

OBSERVATION SPACE
The observation is a ndarray with shape (10, 10, 4) where the channels correspond to the following:

Channel Description
0 player - position of Asterix character
1 enemies - positions of enemy objects
2 treasure - positions of collectible treasure
3 background - static background elements

Each channel contains binary values (0 or 1) indicating presence/absence of the respective element.

ACTION SPACE
The action space consists of 5 discrete actions:

Num Action
0 no-op (no movement)
1 move up
2 move down
3 move left
4 move right

TRANSITION DYNAMICS
- The player can move freely in all four cardinal directions
- Enemies spawn from the sides and move across the screen
- Treasure objects spawn from the sides and move across the screen
- Player must avoid enemies while collecting treasure
- Objects that reach the opposite side disappear

REWARD
- +1 reward for each treasure collected
- No negative rewards

STARTING STATE
- Player starts in a central position
- Initial enemies and treasure spawn from the sides

EPISODE END
The episode ends if either of the following happens:
1. Termination: Player collides with an enemy
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000)
