module MuZero

include("Constructors.jl")

include("SelfPlay.jl")

include("ReplayBuffer.jl")

include("Learning.jl")

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




