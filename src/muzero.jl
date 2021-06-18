module MuZero

include("Constructors.jl")

include("SelfPlay.jl")

include("ReplayBuffer.jl")

include("Learning.jl")



#this should be directly logged with TBL
progress_stats=Dict{String, Float32}(    
	"episode_length" => 0,
    "total_reward" => 0,
    "muzero_reward" => 0,
    "opponent_reward" => 0,
    "mean_value" => 0,
    "total_loss" => 0,
    "value_loss" => 0,
    "reward_loss" => 0,
)

#TODO instead of these dicts, make a log file
progress = Dict{String, Int}(
	"training_step" => 0,
    "num_played_games" => 0,
    "num_played_steps" => 0,
    "num_reanalysed_games" => 0,
	"total_samples" => 0
)

buffer = Dict{Int, GameHistory}() # initial_buffer, used in self-play first to save_game then in replay buffer and learning

# # Workers. But how many cpus to each?
# self_play_workers = nothing
# test_worker = nothing
# training_worker = nothing
# reanalyse_worker = nothing
# replay_buffer_worker = nothing  # these are storages so what part uses CPU?
# shared_storage_worker = nothing  # these are storages so what part uses CPU?

# # run reanalyse on reanalyse_worker #TODO

# # check how to use reserources like multiple cpus and gpus

# function Muzero()
	# # Load the game and the config from the module with the game name
	# # fix the random seed
	# # initialize Checkpoints and replay buffer
	# # run self_play() on self_play_workers
	# # run continuous_update_weight() on training_worker

    #     # Launch the test worker to get performance metrics
    #     # Write everything in TensorBoard
    #         # Persist replay buffer to disk ?
# end
		
# #Search for hyperparameters by launching parallel experiments.
# function hyperparameter_search() end

end




