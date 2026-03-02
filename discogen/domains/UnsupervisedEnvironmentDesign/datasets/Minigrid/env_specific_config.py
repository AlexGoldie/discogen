config = {
    "env_name": "Minigrid",
    "env": {
        "agent_view_size": 5,
        "n_walls": 20,
    },
    "eval": {
        "num_dr_eval_levels": 512,
        "eval_levels": [
            "SixteenRooms",
            "SixteenRooms2",
            "Labyrinth",
            "LabyrinthFlipped",
            "Labyrinth2",
            "StandardMaze",
            "StandardMaze2",
            "StandardMaze3",
        ],
        "eval_num_attempts": 20,
        "eval_freq": 50,
        "EVAL_ON_SAMPLED": True,
    },
}
