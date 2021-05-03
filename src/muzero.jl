module MuZero

#fix the random seed

#check how to use reserources like multiple cpus and gpus

checkpoint = Dict(
    "weights" => nothing,
    "optimizer_state" => nothing,
    "total_reward" => 0,
    "muzero_reward" => 0,
    "opponent_reward" => 0,
    "episode_length" => 0,
    "mean_value" => 0,
    "training_step" => 0,
    "lr" => 0,
    "total_loss" => 0,
    "value_loss" => 0,
    "reward_loss" => 0,
    "policy_loss" => 0,
    "num_played_games" => 0,
    "num_played_steps" => 0,
    "num_reanalysed_games" => 0,
    "terminate" => false,
)

replay_buffer=Dict()

cpu_weights= ("weights","summary") #TODO

# Workers. But how many cpus to each?
self_play_workers = nothing
test_worker = nothing
training_worker = nothing
reanalyse_worker = nothing
replay_buffer_worker = nothing  #these are storages so what part uses CPU?
shared_storage_worker = nothing  #these are storages so what part uses CPU?

# run continuous_self_play() on self_play_workers
# run continuous_update_weight() on training_worker
# run reanalse on reanalyse_worker #TODO



end

