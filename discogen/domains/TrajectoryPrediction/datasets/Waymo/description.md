DESCRIPTION
Waymo Open Dataset (Motion) is a large-scale autonomous driving dataset collected across six US cities (San Francisco, Phoenix, Mountain View, Los Angeles, Detroit, and Seattle) at 10 Hz. It contains over 100,000 segments covering diverse urban and suburban driving conditions. Each Waymo segment has 91 timesteps: 1 second of past (11 timesteps up to and including the current frame, `current_time_index=10`) and 8 seconds of future (80 timesteps). The preprocessed subset used here contains 850 driving scenarios for trajectory prediction. Each sample represents a single focal agent, encoding past motion over 21 timesteps (zero-padded from the native 11 to match the standard format), surrounding traffic (up to 32 agents), and local map geometry as polylines. The prediction horizon is 60 future timesteps (6 seconds, truncated from the native 8 seconds).

OBSERVATION SPACE
Each sample (.pkl) contains:
- obj_trajs: (32, 21, 2) - past positions of up to 32 agents over 21 timesteps (in focal-agent-centric frame)
- obj_trajs_mask: (32, 21) - validity mask (first 10 timesteps are zero-padded for Waymo)
- map_polylines: (128, 20, 2) - lane geometry encoded as polylines
- map_polylines_mask: (128, 20) - validity mask
- track_index_to_predict: index of the focal agent in obj_trajs

TARGET SPACE
- center_gt_trajs: (60, 2) - ground truth future trajectory (x, y) in focal-agent-centric frame
- center_gt_trajs_mask: (60,) - validity mask for each future timestep

EVALUATION METRICS
Models are evaluated using four metrics computed over K predicted trajectory modes:
- minADE: minimum Average Displacement Error across modes (average L2 over valid timesteps for the best mode)
- minFDE: minimum Final Displacement Error across modes (L2 at the last valid timestep for the best mode)
- Miss Rate: fraction of samples where minFDE exceeds 2.0 meters
- Brier-minFDE: minFDE plus (1 - probability of the best mode), penalizing low-confidence predictions

DATASET STRUCTURE
Total: 850 scenarios
Train: 680 (80%)
Test: 170 (20%)

TIMING NOTES
- Native Waymo: 91 timesteps @ 10 Hz (9.1 s total), current_time_index=10
- The 11-step past history is zero-padded to 21 steps to match the standard format (first 10 entries of obj_trajs_mask are 0)
- Future is truncated from 80 to 60 timesteps (6 s), matching nuScenes and Argoverse2 conventions
