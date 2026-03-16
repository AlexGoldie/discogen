DESCRIPTION
Argoverse 2 is a large-scale motion forecasting dataset collected across six US cities (Austin, Detroit, Miami, Pittsburgh, Palo Alto, and Washington D.C.) using high-definition LiDAR and HD maps. It features diverse driving scenarios including highway merging, unprotected turns, and dense urban traffic. The preprocessed subset used here contains 850 driving scenarios for trajectory prediction, where each scenario captures a focal agent's past motion (21 timesteps at 10 Hz), surrounding traffic participants (up to 32 agents), and local road geometry encoded as map polylines. The prediction horizon is 60 future timesteps (6 seconds).

OBSERVATION SPACE
Each sample (.pkl) contains:
- obj_trajs: (32, 21, 2) - past positions of up to 32 agents over 21 timesteps
- obj_trajs_mask: (32, 21) - validity mask
- map_polylines: (128, 20, 2) - lane geometry
- map_polylines_mask: (128, 20) - validity mask
- track_index_to_predict: target agent index

TARGET SPACE
- center_gt_trajs: (60, 2) - ground truth future trajectory (x, y positions)
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
