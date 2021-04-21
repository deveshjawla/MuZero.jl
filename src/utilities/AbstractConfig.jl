export Config


using Parameters

@with_kw mutable struct Config
    seed = 0  # Seed for numpy, torch and the game
    max_num_gpus = nothing  # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. nothing will use every GPUs available

    ### Game
    observation_shape = (3, 96, 96)  # Dimensions of the game observation, must be 3D (channel, height, width). For a 1D array, please reshape it to (1, 1, length of array)
    action_space = "list(range(4))"  # Fixed list of all possible actions. You should only edit the length
    players = "list(range(1))"  # List of players. You should only edit the length
    stacked_observations = 32  # Number of previous observations and previous actions to add to the current observation

    # Evaluate
    muzero_player = 0  # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    opponent = nothing  # Hard coded agent that MuZero faces to assess his progress in multiplayer games. It doesn't influence training. nothing, "random" or "expert" if implemented in the Game class

    ### Self-Play
    num_workers = 350  # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu = false
    max_moves = 27000  # Maximum number of moves if game is not finished before
    num_simulations = 50  # Number of future moves self-simulated
    discount = 0.997  # Chronological discount of the reward
    temperature_threshold = nothing  # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If nothing, visit_softmax_temperature_fn is used every time

    # Root prior exploration noise
    root_dirichlet_alpha = 0.25
    root_exploration_fraction = 0.25

    # UCB formula
    pb_c_base = 19652
    pb_c_init = 1.25

    ### Network
    network = "resnet"  # "resnet" / "fullyconnected"
    support_size = 300  # Value and reward are scaled (with almost sqrt) and encoded on a vector with a range of -support_size to support_size. Choose it so that support_size <= sqrt(max(abs(discounted reward)))

    # Residual Network
    downsample = "resnet"  # Downsample observations before representation network, false / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
    blocks = 16  # Number of blocks in the ResNet
    channels = 256  # Number of channels in the ResNet
    reduced_channels_reward = 256  # Number of channels in reward head
    reduced_channels_value = 256  # Number of channels in value head
    reduced_channels_policy = 256  # Number of channels in policy head
    resnet_fc_reward_layers = [256, 256]  # Define the hidden layers in the reward head of the dynamic network
    resnet_fc_value_layers = [256, 256]  # Define the hidden layers in the value head of the prediction network
    resnet_fc_policy_layers = [256, 256]  # Define the hidden layers in the policy head of the prediction network

    # Fully Connected Network
    encoding_size = 10
    fc_representation_layers = []  # Define the hidden layers in the representation network
    fc_dynamics_layers = [16]  # Define the hidden layers in the dynamics network
    fc_reward_layers = [16]  # Define the hidden layers in the reward network
    fc_value_layers = []  # Define the hidden layers in the value network
    fc_policy_layers = []  # Define the hidden layers in the policy network

    ### Training
    results_path = path # Path to store the model weights and TensorBoard logs
    save_model = true  # Save the checkpoInt in results_path as model.checkpoInt
    training_steps = Int(1000e3)  # Total number of training steps (ie weights update according to a batch)
    batch_size = 1024  # Number of parts of games to train on at each training step
    checkpoInt_Interval = Int(1e3)  # Number of training steps before using the model for self-playing
    value_loss_weight = 0.25  # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
    train_on_gpu = "has_cuda()"  # Train on GPU if available

    optimizer = "SGD"  # "Adam" or "SGD". Paper uses SGD
    weight_decay = 1e-4  # L2 weights regularization
    momentum = 0.9  # Used only if optimizer is SGD

    # Exponential learning rate schedule
    lr_init = 0.05  # Initial learning rate
    lr_decay_rate = 0.1  # Set it to 1 to use a constant learning rate
    lr_decay_steps = 350e3

    ### Replay Buffer
    replay_buffer_size = Int(1e6)  # Number of self-play games to keep in the replay buffer
    num_unroll_steps = 5  # Number of game moves to keep for every batch element
    td_steps = 10  # Number of steps in the future to take Into account for calculating the target value
    PER = true  # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    PER_alpha = 1  # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    # Reanalyze (See paper appendix Reanalyse)
    use_last_model_value = true  # Use the last model to provide a fresher, stable n-step value (See paper appendix Reanalyze)
    reanalyse_on_gpu = false

    ### Adjust the self play / training ratio to avoid over/underfitting
    self_play_delay = 0  # Number of seconds to wait after each played game
    training_delay = 0  # Number of seconds to wait after each training step
    ratio = nothing  # Desired training steps per self played step ratio. Equivalent to a synchronous version, training can take much longer. Set it to nothing to disable it
end