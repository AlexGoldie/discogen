DESCRIPTION
SMAX is a purely JAX-based SMAC (StarCraft Multi-Agent Challenge)-like environment for decentralised unit micromanagement. Two teams of units engage on a 32×32 map: one team of ally agents (learned policies) and one team of enemies (built-in heuristic). Each agent step advances the environment by 8 internal physics steps, each 1/16 of a second, giving smooth unit movement and combat resolution. Units can move, stop, or attack enemies within range. The objective is to eliminate all enemy units before losing the entire allied team.

MULTI AGENT ENVIRONMENT
One agent is assigned to each allied unit. Agents act simultaneously and observe only their own local egocentric view; observations are not shared between agents. Enemy units are controlled by the HeuristicEnemySMAX policy, which attacks the closest ally. Because each agent acts from partial information, coordination must emerge entirely from the learned joint policy.

OBSERVATION SPACE
The observation type is unit_list. Each agent receives a flat vector of shape (218,) built from three groups of features:

Own unit (10 features): normalised health (health / max_health), normalised x/y position (pos / map_size), weapon cooldown, unit type as a 6-bit one-hot vector
Each other allied unit (13 features each, 7 entries): normalised health, relative x/y position normalised by the observing unit's sight range, previous movement vector (x, y), previous attack target index, weapon cooldown, unit type one-hot
Each enemy unit (13 features each, 9 entries): same 13-feature structure; previous actions are visible because see_enemy_actions=True

Units outside the observing unit's sight range appear with all-zero features. All values are continuous real numbers.
Total observation size: 13×(8−1) + 13×9 + 10 = 218

ACTION SPACE
Each agent selects one discrete action per step:

0–3: move in four directions
4: stop (also used as no-op by dead units)
5–13: attack enemy unit 0 through 8

Total actions per ally agent: 14 (5 movement/stop + 9 attack targets). Unavailable actions (e.g. attacking a unit out of range, moving when dead) are masked each step (GET_AVAIL_ACTIONS=True).

TRANSITION DYNAMICS
Each agent step runs 8 internal world steps. In each world step, all units act simultaneously: each unit either moves in its chosen direction at its type-specific velocity (scaled by time_per_step = 1/16 s) or attacks a target. A unit can move or attack but not both in the same world step. An attack succeeds if the target is within attack range and the attacker's weapon cooldown is ≤ 0; the cooldown is then reset to the unit's type-specific value plus a small random deviation. All damage is accumulated and applied simultaneously across the world step. Overlapping units are pushed apart after each world step. A unit touching the map boundary (≤ 0 or ≥ 32) dies immediately (walls_cause_death=True). Dead units are removed and their agent is forced to issue stop.

REWARD
All allied agents share the same team reward computed each step:

Dense reward: sum of normalised enemy health decreases this step, divided by the number of enemy units (proportional to damage dealt)
Win bonus: +1.0 when all enemy units are eliminated and at least one ally survives

There is no explicit loss penalty. Dead agents still receive the team reward to allow for sacrificial play.

STARTING STATE
Allied units are placed near position (8, 16) on the 32×32 map with uniform random noise in [−2, +2] on each axis. Enemy units are placed near (24, 16) with the same noise. All units start with full health and zero weapon cooldown.

EPISODE END
Win: all enemy units are eliminated and at least one ally is alive
Loss: all allied units are eliminated
Truncation: episode length reaches max_steps = 100 agent steps

SCENARIO
3 stalkers and 5 zealots vs 3 stalkers and 6 zealots. An asymmetric version of 3s5z where the enemy team has one additional zealot, making the allied team outnumbered 8 to 9. Stalkers are ranged (attack range 6.0, health 160, damage 13.0, cooldown 1.87 s) and zealots are melee (attack range 2.0, health 150, damage 8.0, cooldown 0.86 s). The extra enemy zealot gives the enemy a larger melee frontline, increasing the pressure on allied zealots and reducing the margin for error in target selection. Allies must apply the same role differentiation as 3s5z but with greater damage efficiency to compensate for being outnumbered. Total allied agents: 8. Observation size: 218. Actions per agent: 14.
