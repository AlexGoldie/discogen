config = {
    "env_name": "Kinetix",
    "env_size": {
        "num_polygons": 5,
        "num_circles": 2,
        "num_joints": 1,
        "num_thrusters": 1,
        "env_size_name": "s",
        "num_motor_bindings": 4,
        "num_thruster_bindings": 2,
        "env_size_type": "predefined",
    },
    "eval": {
        "num_dr_eval_levels": 512,
        "eval_levels": [
            "s/h0_weak_thrust",
            "s/h7_unicycle_left",
            "s/h3_point_the_thruster",
            "s/h4_thrust_aim",
            "s/h1_thrust_over_ball",
            "s/h5_rotate_fall",
            "s/h9_explode_then_thrust_over",
            "s/h6_unicycle_right",
            "s/h8_unicycle_balance",
            "s/h2_one_wheel_car",
        ],
        "eval_num_attempts": 20,
        "eval_freq": 50,
        "EVAL_ON_SAMPLED": True,
    },
    "model": {
        "fc_layer_depth": 3,
        "fc_layer_width": 256,
        "separate_actor_critic": False,
        "activation": "tanh",
        "recurrent_model": False,
        "permutation_invariant_mlp": True,
    },
    "env": {
        "action_type": "multi_discrete",
        "observation_type": "symbolic_flat",
        "dense_reward_scale": 0.2,
        "frame_skip": 2,
        "add_shape_info_to_thruster_joints": False,
    },
    "learning": {
        "filter_levels": True,
        "level_filter_n_steps": 64,
        "level_filter_sample_ratio": 2,
    },
    "train_levels":{
        "train_level_mode": "random"
    }
}
