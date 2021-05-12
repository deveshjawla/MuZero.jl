using Parameters:@with_kw

@with_kw struct GeneralParams
    seed::Int64 = 1337
    max_num_gpus::Int64 = 0 # Fix the maximum number of GPUs to use. It's usually faster to use a single GPU (set it to 1) if it has enough memory. None will use every GPUs available
    observation_shape::Tuple{Int64,Int64,Int64} = (20, 20, 3) # Dimensions of the game observation, must be 3D (WHC). For a 1D array, please reshape it to (1, 1, length of array)
    stacked_observations::Int64 = 32 # Number of previous observations and previous actions to add to the current observation
    muzero_player::Int64 = 0 # Turn Muzero begins to play (0: MuZero plays first, 1: MuZero plays second)
    opponent::String = "expert" # Hard coded agent that MuZero faces to assess its progress in multiplayer games. It doesn't influence training. None, "random" or "expert" if implemented in the Game class
end

@with_kw struct SelfPlayParams
    num_workers::Int64 = 350 # Number of simultaneous threads/workers self-playing to feed the replay buffer
    selfplay_on_gpu::Bool = false
    max_moves::Int64 # Game dependent
    temperature_threshold::Union{Int64,Nothing} = nothing # Number of moves before dropping the temperature given by visit_softmax_temperature_fn to 0 (ie selecting the best action). If None, visit_softmax_temperature_fn is used every time
end

@with_kw struct MCTSParams
    dirichlet_α::Float64 = 0.25
    exploration_ϵ::Float64 = 0.25
    pb_c_base::Int64 = 19652
    pb_c_init::Float64 = 1.25
    discount::Float64 = 0.997 # Chronological discount of the reward
    num_iters::Int64 = 50 # Number of future moves self-simulated
end

@with_kw struct ReplayBufferParams
    replay_buffer_size::Int64 # Number of self-play games to keep in the replay buffer
    num_unroll_steps::Int64 # Number of game moves to keep for every batch element
    td_steps::Int64 = 10 # Number of steps in the future to take into account for calculating the target value
    PER::Bool = true # Prioritized Replay (See paper appendix Training), select in priority the elements in the replay buffer which are unexpected for the network
    PER_alpha::Int64 = 1 # How much prioritization is used, 0 corresponding to the uniform case, paper suggests 1
    num_played_games
    num_played_steps
    total_samples
end


@with_kw struct TrainParams
    results_path::String = mkpath("./results")
    save_model::Bool = true # Save the checkpoint in results_path as model.checkpoint
    training_steps::Int64 = 10000 # Total number of training steps (ie weights update according to a batch)
	epochs::Int64
    batch_size::Int64 = 1024 # Number of parts of games to train on at each training step
    checkpoint_Interval::Int64 = 10 # Number of training steps before using the model for self-playing
    value_loss_weight::Float64 = 0.25 # Scale the value loss to avoid overfitting of the value function, paper recommends 0.25 (See paper appendix Reanalyze)
end

@with_kw struct FeedForwardHP
	width::Int
	depth_common::Int
	depth_second_head::Int = 1
	depth_first_head::Int = 1
	use_batch_norm::Bool = false
	batch_norm_momentum::Float32 = 0.6f0
end

@with_kw struct ResNetHP
	num_blocks::Int # Number of blocks in the ResNet
	num_filters::Int
	conv_kernel_size::Tuple{Int,Int}
	num_second_head_filters::Int = 2
	num_first_head_filters::Int = 1
	batch_norm_momentum::Float32 = 0.6f0
	downsample::Bool = false # Downsample observations before representation network, False / "CNN" (lighter) / "resnet" (See paper appendix Network Architecture)
end




