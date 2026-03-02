import flashbax as fbx

def get_replay_buffer(config):
    buffer = fbx.make_flat_buffer(
        max_length=config["BUFFER_SIZE"],
        min_length=config["BUFFER_BATCH_SIZE"],
        sample_batch_size=config["BUFFER_BATCH_SIZE"],
        add_sequences=False,
        add_batch_size=config["NUM_ENVS"],
    )
    return buffer
