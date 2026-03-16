from collections import defaultdict

import jax
import jax.numpy as jnp
import numpy as np


def supply_rng(f, rng=jax.random.PRNGKey(0)):
    """Helper function to split the random number generator key before each call to the function."""

    def wrapped(*args, **kwargs):
        nonlocal rng
        rng, key = jax.random.split(rng)
        return f(*args, seed=key, **kwargs)

    return wrapped


def flatten(d, parent_key='', sep='.'):
    """Flatten a dictionary."""
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if hasattr(v, 'items'):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def add_to(dict_of_lists, single_dict):
    """Append values to the corresponding lists in the dictionary."""
    for k, v in single_dict.items():
        dict_of_lists[k].append(v)


def evaluate(
    actor_def,
    actor_params,
    env,
    config,
    num_episodes=50,
    rng=None,
    temperature=0.0,
):
    """Evaluate the agent in the environment.

    Args:
        actor_def: Actor network definition.
        actor_params: Actor network parameters.
        env: Evaluation environment.
        config: Configuration dictionary.
        num_episodes: Number of evaluation episodes.
        rng: JAX random key for action sampling.
        temperature: Action noise temperature (0 = deterministic).

    Returns:
        A tuple containing (stats, trajs, renders).
    """
    if rng is None:
        rng = jax.random.PRNGKey(np.random.randint(0, 2**32))

    @jax.jit
    def get_action(params, obs, seed):
        """Get action from policy."""
        dist = actor_def.apply({'params': params}, obs, temperature=1.0)
        action = dist.mode()

        if temperature > 0:
            noise = jax.random.normal(seed, action.shape) * config['actor_noise'] * temperature
            noise = jnp.clip(noise, -config['actor_noise_clip'], config['actor_noise_clip'])
            action = action + noise

        return jnp.clip(action, -1, 1)

    # Wrap get_action with supply_rng for cleaner RNG handling
    actor_fn = supply_rng(
        lambda observations, seed, temp: get_action(actor_params, observations, seed),
        rng=rng,
    )

    trajs = []
    stats = defaultdict(list)
    renders = []

    total_episodes = num_episodes
    for i in range(total_episodes):
        traj = defaultdict(list)
        should_render = i >= num_episodes

        observation, info = env.reset()
        done = False
        step = 0
        render = []
        episode_return = 0.0

        while not done:
            action = actor_fn(observations=observation, temp=temperature)
            action = np.array(action)
            action = np.clip(action, -1, 1)

            next_observation, reward, terminated, truncated, info = env.step(action)
            done = terminated or truncated
            step += 1
            episode_return += reward

            if should_render and (step % video_frame_skip == 0 or done):
                frame = env.render().copy()
                render.append(frame)

            transition = dict(
                observation=observation,
                next_observation=next_observation,
                action=action,
                reward=reward,
                done=done,
                info=info,
            )
            add_to(traj, transition)
            observation = next_observation

        if i < num_episodes:
            episode_stats = {
                'episode.return': episode_return,
                'episode.length': step,
            }

            # Add normalized return if environment supports it
            if hasattr(env.unwrapped, 'get_normalized_score'):
                episode_stats['episode.normalized_return'] = (
                    env.unwrapped.get_normalized_score(episode_return) * 100.0
                )

            flattened_info = flatten(info)
            for k, v in flattened_info.items():
                if k not in episode_stats:
                    episode_stats[k] = v

            add_to(stats, episode_stats)
            trajs.append(traj)
        else:
            renders.append(np.array(render))

    for k, v in stats.items():
        stats[k] = np.mean(v)

    return stats, trajs, renders
