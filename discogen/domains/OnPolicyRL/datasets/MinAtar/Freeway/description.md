DESCRIPTION
Freeway MinAtar is a simplified version of the classic Atari Freeway game. The player starts at bottom of screen and can travel up/down. Player speed is restricted s.t. player only moves every 3 frames. Reward +1 given when player reaches the top of the screen. The player must navigate through traffic that moves horizontally across the screen while trying to reach the opposite side.

OBSERVATION SPACE
The observation is a ndarray with shape (10, 10, 4) where the channels correspond to the following:

Channel Description
0 chicken - position of the player character
1 cars - positions of moving cars/traffic
2 trail - indicates recent position or movement
3 background - static background elements

Each channel contains binary values (0 or 1) indicating presence/absence of the respective element.

ACTION SPACE
The action space consists of 3 discrete actions:

Num Action
0 no-op (no movement)
1 move up
2 move down

TRANSITION DYNAMICS
- The player moves up or down based on the chosen action but at reduced speed (every 3 frames)
- Cars move horizontally across the screen at different speeds
- Player must avoid colliding with cars
- When player reaches the top, they receive reward and can continue

REWARD
- +1 reward for reaching the top of the screen
- No negative rewards for collisions

STARTING STATE
- Player starts at the bottom of the screen
- Cars spawn and move across different lanes

EPISODE END
The episode ends if either of the following happens:
1. Termination: Maximum episode length reached
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000)
