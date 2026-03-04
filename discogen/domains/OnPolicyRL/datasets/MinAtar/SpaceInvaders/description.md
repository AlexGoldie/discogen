DESCRIPTION
Space Invaders MinAtar is a simplified version where the player controls a cannon at the bottom of the screen and can shoot bullets upward at a cluster of aliens above. The aliens move across the screen until one of them hits the edge, at which point they all move down and switch directions.

OBSERVATION SPACE
The observation is a ndarray with shape (10, 10, 4) where the channels correspond to the following:

Channel Description
0 cannon - position of the player's cannon
1 aliens - positions of alien enemies
2 bullets - positions of player bullets
3 alien_bullets - positions of enemy bullets

Each channel contains binary values (0 or 1) indicating presence/absence of the respective element.

ACTION SPACE
The action space consists of 4 discrete actions:

Num Action
0 no-op (no movement)
1 move left
2 move right
3 fire bullet

TRANSITION DYNAMICS
- The cannon moves left or right based on action
- Player can fire bullets upward
- Aliens move horizontally and drop down when reaching edges
- Aliens occasionally fire bullets downward
- Bullets destroy aliens on contact
- Player loses if hit by alien bullet

REWARD
- +1 reward for each alien destroyed
- No negative rewards

STARTING STATE
- Cannon starts at bottom center
- Grid of aliens positioned in upper portion
- No bullets initially present

EPISODE END
The episode ends if either of the following happens:
1. Termination: Player cannon is hit by alien bullet or aliens reach the bottom
2. Truncation: The length of the episode reaches max_steps_in_episode (default: 1000)
