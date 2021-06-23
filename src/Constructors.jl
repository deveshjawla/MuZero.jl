using Parameters:@with_kw

@with_kw struct Config
    seed::Int = 1337
    observation_shape::Tuple{Int,Int,Int} # Dimensions of the game observation, must be 3D (WHC). For a 1D array, please reshape it to (1, 1, length of array)
    action_space
	players
	stacked_observations::Int # Number of previous observations and previous actions to add to the current observation
    muzero_player::Int = 1 # Turn Muzero begins to play (1: MuZero plays first, 2: MuZero plays second)
    opponent::String = "expert" # Hard coded agent that MuZero faces to assess its progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
	intermediate_rewards::Bool = false

    num_workers::Int # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu::Bool = false
    max_moves::Int # Game dependent
    temperature_threshold::Union{Int,Nothing} = nothing # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time

    dirichlet_α::Float32 = 0.25
    exploration_ϵ::Float32 = 0.25
    pb_c_base::Int = 19652
    pb_c_init::Float32 = 1.25
    discount::Float32 = 0.997 # Chronological discount of the reward
    num_iters::Int # Number of future moves self-simulated

    replay_buffer_size::Int = 10000 # Number of self-play games to keep in the replay buffer
    num_unroll_steps::Int # Number of game moves to keep for every batch element
    td_steps::Int # Number of steps in the future to take into account for calculating the target value
    PER::Bool # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    PER_alpha::Int = 1 # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1

    results_path::String = mkpath("./results")
    networks_path::String = mkpath("./networks")
    training_steps::Int # Total number of training steps (ie weights update according to a batch)
    batch_size::Int # Number of parts of games to train on at each training step
    checkpoint_interval::Int = 10 # Number of training steps before using the model for self-playing
    value_loss_weight::Float32 = 0.25 # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
end

"""
Stores info about the Buffer, gets modified by the sample_game and save_game functions
"""
mutable struct BufferStats #TODO use better way of logging for eg. log to file
	num_played_games::Int
	num_played_steps::Int
	num_reanalysed_games::Int
	total_samples::Int
end


mutable struct LearningProgress
	training_step::Int
end

@with_kw struct FeedForwardHP
	width_hidden::Int
	depth_representation::Int
	depth_prediction::Int
	depth_dynamics::Int
	depth_policy::Int
	depth_value::Int
	depth_reward::Int
	depth_state_head::Int
	use_batch_norm::Bool = false
	batch_norm_momentum::Float32 = 0.6f0
	hidden_state_size::Int #should be equal to prod(observation_shape)
	reward_activation
end

@with_kw mutable struct ResNetHP
	num_blocks::Int # Number of blocks in the ResNet
	depth_representation::Int
	num_filters::Int
	conv_kernel_size::Tuple{Int,Int}
	num_second_head_filters::Int = 2
	num_first_head_filters::Int = 1
	batch_norm_momentum::Float32 = 0.6f0
	downsample::Bool = false
	hidden_state_size::Int
	representation_output_size
	depth_policy::Int
	depth_value::Int
end




