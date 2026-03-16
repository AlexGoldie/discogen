Offline reinforcement learning, also known as batch reinforcement learning, is a paradigm where agents learn policies from fixed datasets of previously collected trajectories without any online environment interaction. Unlike online RL where agents explore and learn simultaneously, offline RL must extract effective policies purely from pre-existing data, which may have been collected by different policies at varying levels of expertise.

This paradigm is particularly valuable when online interaction is expensive, risky, or impractical—such as in robotics, healthcare, or autonomous systems where trial-and-error exploration could be costly or dangerous or not feasible. The agent must learn to make decisions in states that may differ from those in the dataset, while avoiding overestimation of Q-values for out-of-distribution actions.

Common algorithmic components used in offline reinforcement learning are:

1. **Deeper Newtwors**: Increase the number of hidden layers in both the actor and the critic network
2. **Layer Norm**: Apply Layer Norms to the critic network
3. **Actor and Critic Penalty Decoupling**: Penalize the actor and critic using different hyperparameters that are tuned separately and are specified in the config of the algorithm
4. **Discount Factor Change**: The value of gamma can have profound effects on the online evaluation performance. Different datasets and tasks have different gammas


Your task is to implement the core components of an offline RL algorithm. The training infrastructure, environment interface, and dataset handling are provided. You should focus on:
- **actor_loss.py**: The actor loss function
- **critic_loss.py**: The critic loss function
- **networks.py**: The neural network architectures
- **optim.py**: The optimizer configuration
- **train.py**: The training loop and target updates

The algorithm will be evaluated on OGBench environments including navigation tasks (AntMaze, HumanoidMaze) and manipulation tasks (Cube, Scene, Puzzle). Success is measured by the normalized return achieved by the learned policy.

Below, we provide descriptions of the specific environments and datasets you will be training on. Note that any algorithm you develop may be applied to additional OGBench environments for testing generalization.
