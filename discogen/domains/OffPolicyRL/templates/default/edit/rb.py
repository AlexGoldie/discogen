import flashbax as fbx

def get_replay_buffer(config):
    """
    Replay Buffer Implementation Guide
    ===================================

    Implement a stateless replay buffer for reinforcement learning with JAX compatibility.

    OVERVIEW
    --------
    Create a replay buffer object that stores and samples transitions for training.
    The buffer must be stateless (no internal mutable state) and JIT-compatible.

    INPUTS
    ------
    - config: Configuration dict from `config.py` containing hyperparameters. Definately includes BUFFER_SIZE, BUFFER_BATCH_SIZE and NUM_ENVS.

    RETURNS
    -------
    - buffer: A stateless object with replay buffer operations (see Methods below)

    REQUIRED PROPERTIES
    -------------------
    - buffer_size: Maximum capacity of the replay buffer
    - buffer_batch_size: Batch size for sampling
    - buffer_state: External state representing buffer contents (not stored internally)

    REQUIRED METHODS
    ----------------
    1. init(timestep) -> buffer_state
    Initialize an empty buffer state with the given timestep structure

    2. add(buffer_state, timestep) -> buffer_state
    Add a new transition to the buffer and return updated state

    3. sample(buffer_state, rng) -> batch
    Sample a batch of experiences from the buffer
    Returns: Batch with structure (first, second) where:
        - first: (obs, action, reward, done)  # Current timestep
        - second: (obs, action)                # Next timestep

    4. can_sample(buffer_state) -> bool
    Check if buffer contains enough samples for a batch

    USAGE EXAMPLE
    -------------
    # 1. Create buffer
    buffer = get_replay_buffer(config)

    # 2. Make it JIT-compatible (required for performance)
    buffer = buffer.replace(
        init=jax.jit(buffer.init),
        add=jax.jit(buffer.add, donate_argnums=0),
        sample=jax.jit(buffer.sample),
        can_sample=jax.jit(buffer.can_sample),
    )

    # 3. Initialize buffer state
    @chex.dataclass(frozen=True)
    class TimeStep:
        obs: chex.Array
        action: chex.Array
        reward: chex.Array
        done: chex.Array

    _timestep = TimeStep(obs=obs, action=action, reward=reward, done=done)
    buffer_state = buffer.init(_timestep)

    # 4. Add transitions during environment interaction
    buffer_state = buffer.add(buffer_state, timestep)

    # 5. Check if ready to sample
    can_sample = buffer.can_sample(buffer_state)

    # 6. Sample and extract batch for training
    if can_sample:
        batch = buffer.sample(buffer_state, rng).experience
        current_step, next_step = batch.first, batch.second
        # Use (current_step, next_step) for training

    IMPORTANT NOTES
    ---------------
    - The buffer MUST be stateless - all state is passed explicitly
    - All methods must be JIT-compatible for performance
    - Use donate_argnums=0 for add() to optimize memory usage
    - Ensure buffer_batch_size <= buffer_size
    """

    class ReplayBuffer:
        # ... implementation here ...
        ...

    return ReplayBuffer # type: ignore
