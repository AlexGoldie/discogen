def make_schedule_fn(config, lr):
    # lr is negative to implement gradient DESCENT under optax syntax.
    def schedule_fn(count):
        frac = (
            1.0
            - (count // (config["NUM_MINIBATCHES"] * config["UPDATE_EPOCHS"]))
            / config["NUM_UPDATES"]
        )
        return lr * frac

    return schedule_fn
