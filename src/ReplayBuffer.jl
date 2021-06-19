"""
The value target is the discounted root value of the search tree td_steps into the
future, plus the discounted sum of all rewards until then.
"""
function compute_target_value(conf::Config, history::GameHistory, index::Int)::Float32
    bootstrap_index = index + conf.td_steps
    if bootstrap_index < length(history.root_values)
        root_values = (isnothing(history.reanalysed_predicted_root_values) ? history.root_values : history.reanalysed_predicted_root_values )
        last_step_value = (history.to_play_history[bootstrap_index] == history.to_play_history[index] ? root_values[bootstrap_index] : -root_values[bootstrap_index] )
        value = last_step_value * conf.discount^conf.td_steps
		for (i, reward) in enumerate(history.reward_history[index:bootstrap_index])
			# The value is oriented from the perspective of the current player
			value += (history.to_play_history[index] == history.to_play_history[index + i] ? reward : -reward) * conf.discount^i
		end
    else
        value = 0.0f0
    end
	# @info "compute_target_value" index bootstrap_index conf.td_steps size(history.reward_history)
    return value
end

"""
Generate targets for every unroll steps.
"""
function make_target(conf::Config, history::GameHistory, state_index::Int)::Tuple{Vector{Float32},Vector{Float32},Array{Float32,2},Vector{Int}}
    target_values, target_rewards, target_policies, actions = Vector{Float32}(), Vector{Float32}(), Matrix{Float32}(undef, length(conf.action_space), 0), Vector{Int}()
	# @info "make_target" state_index conf.num_unroll_steps
    for current_index = state_index:state_index + conf.num_unroll_steps # returns vectors of length num_unroll_steps+1
        if current_index < length(history.root_values)
			value = compute_target_value(conf, history, current_index)
            append!(target_values, value)
            append!(target_rewards, history.reward_history[current_index])
            target_policies = hcat(target_policies, history.child_visits[:, current_index])
            append!(actions, history.action_history[current_index])
        elseif current_index == length(history.root_values)
            append!(target_values, 0)
            append!(target_rewards, history.reward_history[current_index])
            # uniform policy
            target_policies = hcat(target_policies, fill(1.0f0 / length(conf.action_space), length(conf.action_space)))
            append!(actions, history.action_history[current_index])
        else
            # States past the end of games are treated as absorbing states
            append!(target_values, 0)
            append!(target_rewards, 0)
            target_policies = hcat(target_policies, fill(1.0f0 / length(conf.action_space), length(conf.action_space)))
            append!(actions, rand(rng, conf.action_space))
        end
    end
    return target_values, target_rewards, target_policies, actions
end


"""
Used in Reanalyze
"""
function update_history!(conf::Config, history::GameHistory, buffer::Dict{Int,GameHistory}, game_id::Int)::Nothing
    # The element could have been removed since its selection and training
	# Checks if the game_id still in buffer, where buffer keys are sorted Ints
	if first(keys(buffer)) ≤ game_id
		if conf.PER
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
function sample_position(conf::Config, history::GameHistory, force_uniform=false)::Tuple{Int,Float32}
    position_prob = 0.0f0
    if conf.PER && !force_uniform
        position_probs = history.priorities ./ sum(history.priorities)
        position_index = rand(rng, Categorical(position_probs))
        position_prob = position_probs[position_index]
    else
        position_index = rand(1:length(history.root_values))
    end
	# @info "sample_position" length(history.root_values) position_index
    return position_index, position_prob
end

function sample_n_games(conf::Config, buffer::Dict{Int,GameHistory}, force_uniform=false)::Vector{Tuple{Int,GameHistory,Float32}}
    if conf.PER && !force_uniform
        game_id_list = Vector{Int}()
        game_probs = Vector{Float32}()
        for (game_id, history) in buffer
            append!(game_id_list, game_id)
            push!(game_probs, history.game_priority)
        end
        game_probs ./= sum(game_probs)
        game_prob_dict = Dict(game_id => prob for (game_id, prob) in zip(game_id_list, game_probs))
        selected_games = [game_id_list[i] for i in rand(rng, Categorical(game_probs), conf.batch_size)]
		n_games = [(game_id, buffer[game_id], game_prob_dict[game_id]) for game_id in selected_games]
    else
        selected_games = rand(collect(keys(buffer)), conf.batch_size)
        game_prob_dict = Dict()
		n_games  = [(game_id, buffer[game_id], 0.0f0) for game_id in selected_games]
    end
    return n_games
end

function sample_game(conf::Config, buffer::Dict{Int,GameHistory}, force_uniform=false)::Tuple{Int,GameHistory,Float32}
    game_prob = 0.0f0
    if conf.PER && !force_uniform
        game_probs = Vector{Float32}()
        for (_, history) in buffer
            append!(game_probs, history.game_priority)
        end
        game_probs ./= sum(game_probs)
        game_index = rand(rng, Categorical(game_probs))
        game_prob = game_probs[game_index]
    else
        game_index = rand(1:length(buffer))
    end
    game_id = progress["num_played_games"] - length(buffer) + game_index
    return game_id, buffer[game_id], game_prob
end

"""
Saves history in buffer
When using PER, it sets the initial priorities in history
"""
function save_game(conf::Config, history::GameHistory, progress::Dict{String,Int}, buffer::Dict{Int,GameHistory})
    if conf.PER
		# Initial priorities for the prioritized replay (See paper appendix Training)
		priorities = Vector{Float32}()
		for (i, root_value) in enumerate(history.root_values)
			priority =  abs(root_value - compute_target_value(conf, history, i))^conf.PER_alpha
			append!(priorities, priority)
		end
		history.priorities = priorities
		history.game_priority = maximum(history.priorities)
	end
	buffer[progress["num_played_games"]] = history
	progress["num_played_games"] += 1
	progress["num_played_steps"] += length(history.root_values)
	progress["total_samples"] += length(history.root_values)

	if conf.replay_buffer_size < length(buffer)
		del_id = progress["num_played_games"] - length(buffer)
		progress["total_samples"] -= length(buffer[del_id].root_values)
		delete!(buffer, del_id)
	end
end


"""
Used in Learning to Update game and position priorities with priorities calculated during the training.
See Distributed Prioritized Experience Replay (https://arxiv.org/abs/1803.00933)
"""
function update_priorities!(buffer::Dict{Int,GameHistory}, priorities::Matrix{Float32}, index_batch::Vector{Tuple})::Nothing
    for i = 1:length(index_batch)
        game_id, game_pos = index_batch[i]

        # checks if game_id still in buffer
		if first(keys(buffer)) ≤ game_id
			# Update position priorities
			priority = priorities[:,i]
			start_index = game_pos
			end_index = minimum(game_pos + length(priority), length(buffer[game_id].priorities))
			# Update game priorities
			buffer[game_id].priorities[start_index:end_index] = priority[1:end_index - start_index + 1]
			buffer[game_id].game_priority = maximum(buffer[game_id].priorities)
		end
    end
end


function get_batch(conf::Config, buffer::Dict{Int,GameHistory})::Tuple{Vector{Tuple{Int64,Float32}},Tuple{Array{Float32,4},Matrix{Float32},Matrix{Float32},Matrix{Float32},Array{Float32,3},Any,Vector{Float32}}}
    total_samples = sum([length(history.root_values) for history in values(buffer)])
    # @info "Replay buffer initialized with $(total_samples) samples"
    index_batch = Vector{Tuple{Int,Float32}}()
	observation_batch = Array{Float32}(undef, conf.observation_shape[1],conf.observation_shape[2], (conf.observation_shape[3]*(conf.stacked_observations+1)+conf.stacked_observations), 0)
	action_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	reward_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	value_batch = Array{Float32}(undef, conf.num_unroll_steps+1, 0)
	policy_batch = Array{Float32}(undef, length(conf.action_space), conf.num_unroll_steps+1, 0)
	gradient_scale_batch = Vector{Float32}()
    weight_batch = conf.PER ? Vector{Float32}() : nothing
	n_games= sample_n_games(conf, buffer) # makes a batch of (batch_size) game unrolls
	# @info "Sampled N games" size(n_games)
    for (game_id, history, game_prob) in n_games 
        game_pos, pos_prob = sample_position(conf, history)
		# @info "get_batch"  size(history.observation_history) size(history.action_history) size(history.reward_history) size(history.to_play_history) size(history.child_visits) size(history.root_values)
        target_values, target_rewards, target_policies, actions = make_target(conf, history, game_pos) # unrolls each game sample for num_unroll_steps
        push!(index_batch, (game_id, game_pos))
		# @info "Empty batch and Target sizes are:" size(observation_batch) size(action_batch)  size(reward_batch) size(value_batch ) size(policy_batch) size(gradient_scale_batch) size(target_values) size(target_rewards) size(target_policies) size(actions)
        observation_batch = cat(observation_batch, get_stacked_observations(conf, history, game_pos, conf.stacked_observations), dims=ndims(observation_batch))
        action_batch = hcat(action_batch, actions)
        reward_batch = hcat(reward_batch, target_rewards)
        value_batch = hcat(value_batch, target_values)
        policy_batch = cat(policy_batch, target_policies, dims=ndims(target_policies) + 1)
        push!(gradient_scale_batch, minimum([conf.num_unroll_steps, length(history.action_history)+1 - game_pos]))
        conf.PER ? push!(weight_batch, 1 / (total_samples * game_prob * pos_prob)) : nothing
    end
    conf.PER ? weight_batch ./= maximum(weight_batch) : nothing
    return index_batch, (observation_batch, action_batch, value_batch, reward_batch, policy_batch, weight_batch, gradient_scale_batch)
end
