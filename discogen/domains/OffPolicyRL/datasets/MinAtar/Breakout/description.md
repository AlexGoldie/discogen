DESCRIPTION
Breakout MinAtar is a simplified version of the classic Atari Breakout game. The player controls a paddle at the bottom of the screen and must bounce a ball to break rows of bricks at the top. The ball travels only along diagonals and bounces off when hitting the paddle or walls. The game continues until the ball hits the bottom of the screen or the maximum number of steps is reached.

OBSERVATION SPACE
The observation is a ndarray with shape (10, 10, 4) where the channels correspond to the following:

Channel   Description
0       paddle - position of the player's paddle
1       ball - current position of the ball
2       trail - indicates the ball's direction of movement
3       brick - layout of the remaining bricks

Each channel contains binary values (0 or 1) indicating presence/absence of the respective element.

ACTION SPACE
The action space consists of 3 discrete actions:

Num    Action
0     no-op (no movement)
1     move paddle left
2     move paddle right

TRANSITION DYNAMICS
- The paddle moves left or right based on the chosen action
- The ball moves diagonally and bounces off walls and the paddle
- When the ball hits a brick, the brick is destroyed
- When all bricks are cleared, a new set of three rows is added
- The ball's direction is indicated by the trail channel

REWARD
- +1 reward for each brick broken
- No negative rewards

STARTING STATE
- Paddle starts at position 4
- Ball starts at either (3,0) or (3,9) with corresponding diagonal direction
- Three rows of bricks are initialized at the top (rows 1-3)

EPISODE END
The episode ends if either of the following happens:
1. Termination: The ball hits the bottom of the screen
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000)

STATE SPACE
The state consists of:
- ball_y: vertical position of ball (0-9)
- ball_x: horizontal position of ball (0-9)
- ball_dir: direction of ball movement (0-3)
- pos: paddle position (0-9)
- brick_map: 10x10 binary map of bricks
- strike: boolean indicating if ball hit something
- last_y, last_x: previous ball position
- time: current timestep
- terminal: whether episode has ended
