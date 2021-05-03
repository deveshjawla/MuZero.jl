include("./networks/SharedStorage.jl")
struct ReplayBufferParams
    replay_buffer_size::Int64 = 1000000
    num_unroll_steps::Int64 = 5
    td_steps::Int64 = 10
    PER::Bool = true
    PER_alpha::Int64 = 1
    num_played_games
    num_played_steps
    total_samples
end

"""
The value target is the discounted root value of the search tree td_steps into the
future, plus the discounted sum of all rewards until then.
"""
function compute_target_value(rbp::ReplayBufferParams, history::GameHistory, index::Int64)::Float64
    bootstrap_index = index + rbp.td_steps
    if bootstrap_index < length(history.root_values)
        root_values = (
            history.reanalysed_predicted_root_values === nothing ? history.root_values :
            history.reanalysed_predicted_root_values
        )
        last_step_value = (
            history.to_play_history[bootstrap_index] == history.to_play_history[index] ?
            root_values[bootstrap_index] : -root_values[bootstrap_index]
        )
        value = last_step_value * config.discount^rbp.td_steps
    else
        value = 0
    end

    for (i, reward) in enumerate(history.reward_history[index:bootstrap_index])
        # The value is oriented from the perspective of the current player
        value +=
            (
                history.to_play_history[index] == history.to_play_history[index+i] ?
                reward : -reward
            ) * config.discount^i
    end
    return value
end

"""
Generate targets for every unroll steps.
"""
function make_target(
    rbp::ReplayBufferParams,
    config::Config,
    history::GameHistory,
    state_index::Int64
)::Tuple{Vector,Vector,Vector,Vector}
    target_values, target_rewards, target_policies, actions = [], [], [], []
    for current_index = state_index:state_index+rbp.num_unroll_steps
        value = compute_target_value(config, history, current_index)

        if current_index < length(history.root_values)
            append!(target_values, value)
            append!(target_rewards, history.reward_history[current_index])
            append!(target_policies, history.child_visits[current_index])
            append!(actions, history.action_history[current_index])
        elseif current_index == length(history.root_values)
            append!(target_values, 0)
            append!(target_rewards, history.reward_history[current_index])
            #uniform policy
            append!(
                target_policies,
                [
                    1 / length(history.child_visits[1]) for
                    _ = 1:length(history.child_visits[1])
                ],
            )
            append!(actions, history.action_history[current_index])
        else
            # States past the end of games are treated as absorbing states
            append!(target_values, 0)
            append!(target_rewards, 0)
            append!(
                target_policies,
                [
                    1 / length(history.child_visits[1]) for
                    _ = 1:length(history.child_visits[1])
                ],
            )
            append!(actions, rand(rng, config.action_space))
        end
    end
    return target_values, target_rewards, target_policies, actions
end


"""
Update game and position priorities with priorities calculated during the training.
See Distributed Prioritized Experience Replay (https://arxiv.org/abs/1803.00933)
"""
function update_priorities!(buffer::Dict, priorities::Array, index_info::Vector{Tuple})::nothing
    keys_buffer = collect(keys(buffer))
    for i = 1:length(index_info)
        game_id, game_pos = index_info[i]

        # The element could have been removed since its selection and training
        if keys_buffer[i] ≤ game_id #TODO
            # Update position priorities
            priority = priorities[i,:]
            start_index = game_pos
            # end_index =
            #     minimum(game_pos + length(priority), length(buffer[game_id].priorities))
            # buffer[game_id].priorities[start_index:end_index] =
            #     priority[1:end_index-start_index] #TODO buffer is a dict not a struct
            # # Update game priorities
            # buffer[game_id].game_priority = maximum(buffer[game_id].priorities)
        end
    end
end

function update_history!(rbp::ReplayBufferParams, history::GameHistory, buffer::Dict, game_id::Int64)::nothing
    # The element could have been removed since its selection and training
    if keys_buffer[i] ≤ game_id #TODO
        if rbp.PER
            # Avoid read only array when loading replay buffer from disk
            history.priorities = copy(history.priorities)
        end
        buffer[game_id] = history
    end
end

"""
Sample position from game either uniformly or according to some priority.
See paper appendix Training.
"""
function sample_position(rbp::ReplayBufferParams, history::GameHistory, force_uniform=false)::Tuple{Int64,Float64}
    position_prob = nothing
    if rbp.PER && !force_uniform
        position_probs = history.priorities ./ sum(history.priorities)
        position_index = rand(rng, Categorical(position_probs))
        position_prob = position_probs[position_index]
    else
        position_index = rand(1:length(history.root_values))
    end
    return position_index, position_prob
end

function sample_n_games(rbp::ReplayBufferParams, buffer::Dict, n_games::Int64, force_uniform=false)::Vector{Tuple}
    if rbp.PER && !force_uniform
        game_id_list = []
        game_probs = Float32[]
        for (game_id, history) in buffer
            append!(game_id_list,game_id)
            append!(game_probs,history.game_priority)
        end
        game_probs ./= sum(game_probs)
        game_prob_dict= Dict(game_id => prob for (game_id, prob) in zip(game_id_list, game_probs))
        selected_games = [game_id_list[i] for i in rand(rng, Categorical(game_probs), n_games)]
    else
        selected_games = rand(collect(keys(buffer)),n_games)
        game_prob_dict = Dict()
    end
    return [(game_id, buffer[game_id], game_prob_dict[game_id]) for game_id in selected_games]
end

function sample_game(rbp::ReplayBufferParams, buffer::Dict, force_uniform=false)::Tuple
    game_prob = nothing
    if rbp.PER && !force_uniform
        game_probs = Float32[]
        for (game_id, history) in buffer
            append!(game_probs,history.game_priority)
        end
        game_probs ./= sum(game_probs)
        game_index = rand(rng, Categorical(game_probs))
        game_prob = game_probs[game_index]
    else
        game_index = rand(1:length(buffer))
    end
    game_id = num_played_games - length(buffer) + game_index #TODO
    return game_id, buffer[game_id], game_prob
end


function get_batch(rbp::ReplayBufferParams,config::Config, buffer)::Tuple
    index_batch, observation_batch, action_batch, reward_batch, value_batch, policy_batch, gradient_scale_batch = [], [], [], [], [], [], []
    weight_batch = rbp.PER ? Float32[] : nothing
    for (game_id, history, game_prob) in sample_n_games(config,buffer,config.batch_size)
        game_pos, pos_prob = sample_position(config,history)
        values, rewards, policies, actions = make_target(rbp, config,history, game_pos)
        append!(index_batch,(game_id, game_pos))
        append!(observation_batch,get_stacked_observations(history,game_pos,config.stacked_observations))
        append!(action_batch,actions)
        append!(reward_batch, rewards)
        append!(value_batch,values)
        append!(policy_batch,policies)
        append!(gradient_scale_batch,)
        append!(gradient_scale_batch,repeat(minimum(rbp.num_unroll_steps,length(history.action_history)-game_pos),length(actions)))
        rbp.PER ? append!(weight_batch,1/(total_samples*game_prob*pos_prob)) : nothing #TODO
    end
    weight_batch =  rbp.PER ? weight_batch ./ maximum(weight_batch) : nothing
    return index_batch, (
                observation_batch,
                action_batch,
                value_batch,
                reward_batch,
                policy_batch,
                weight_batch,
                gradient_scale_batch,
            )
end

function save_game(rbp::ReplayBufferParams, history::GameHistory, checkpoint::Dict, shared_storage=false)
    if rbp.PER
        if history.priorities ≠ nothing
                # Avoid read only array when loading replay buffer from disk
            history.priorities = copy(history.priorities)
        else
                # Initial priorities for the prioritized replay (See paper appendix Training)
                priorities = Float32[]
            for (i,root_value) in enumerate(history.root_values)
                priority =  abs(root_value - compute_target_value(rbp, history,i)) ^ rbp.PER_alpha
                append!(priorities,priority)
            end
            history.game_priority = maximum(history.priorities)
        end
        buffer[replay_buffer.num_played_games] = history
        replay_buffer.num_played_games += 1
        replay_buffer.num_played_steps += length(history.root_values)
        replay_buffer.total_samples += length(history.root_values)

        if rbp.replay_buffer_size < length(buffer)
            del_id = replay_buffer.num_played_games - length(buffer)
            replay_buffer.total_samples -= length(buffer[del_id].root_values) #TODO buffer should be a DICT
             delete!(buffer,del_id)
        end

        if shared_storage
            set_info!(checkpoint, "num_played_games",replay_buffer.num_played_games)
            set_info!(checkpoint, "num_played_steps",replay_buffer.num_played_steps)
        end
    end
end