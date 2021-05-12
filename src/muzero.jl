module MuZero

# fix the random seed

checkpoint = Dict(
    "weights" => nothing,
    "total_reward" => 0,
    "muzero_reward" => 0,
    "opponent_reward" => 0,
    "episode_length" => 0,
    "mean_value" => 0,
    "training_step" => 0,
    "total_loss" => 0,
    "value_loss" => 0,
    "reward_loss" => 0,
    "policy_loss" => 0,
    "num_played_games" => 0,
    "num_played_steps" => 0,
    "num_reanalysed_games" => 0,
    "terminate" => false,
)
# Workers. But how many cpus to each?
self_play_workers = nothing
test_worker = nothing
training_worker = nothing
reanalyse_worker = nothing
replay_buffer_worker = nothing  # these are storages so what part uses CPU?
shared_storage_worker = nothing  # these are storages so what part uses CPU?

# run continuous_self_play() on self_play_workers
# run continuous_update_weight() on training_worker
# run reanalyse on reanalyse_worker #TODO


# check how to use reserources like multiple cpus and gpus

end

RepNetHP = FeedForwardHP(width=gp.width, depth_common=gp.fc_representation_layers, depth_first_head=1)
representation_network = init_network(config,  RepNetHP)

DynNetHP = FeedForwardHP(width=gp.width, depth_common=gp.fc_dynamics_layers, depth_first_head=1, depth_second_head=fc_reward_layers)
dynamics_network = init_network(config,  DynNetHP)

PredNetHP = FeedForwardHP(width=gp.width, depth_common=gp.fc_policy_layers, depth_first_head=gp.fc_value_layers, depth_second_head=gp.fc_policy_layers)
prediction_network = init_network(config,  PredNetHP)


RepNetHP = ResNetHP() # TODO figure out a way to initilise these from the config file
representation_network = init_network(config,  RepNetHP)

DynNetHP = ResNetHP()
dynamics_network = init_network(config,  DynNetHP)

PredNetHP = ResNetHP()
prediction_network = init_network(config,  PredNetHP)

