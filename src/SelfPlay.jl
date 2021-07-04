using BSON
using Serialization
using ReinforcementLearningBase
"""
Makes a single input from a pair of state and action, for dynamics network
"""
function make_state_action(state::Array{Float32,3}, action::Int, conf::Config)::Array{Float32,3}
	action /= length(conf.action_space)
	action = action .* ones(Float32, (conf.observation_shape[1], conf.observation_shape[2]))
	# Scale the gradient by half at the start of the dynamics function (See paper appendix Training)
	state .*= 2.0f0
	state_action = cat(state, action, dims=3)
	return state_action
end
"""
# Holds the Min-Max values of the tree
Simply brings variables('min' and 'max') and functions('update' and 'normalize')
to the scope.
'update_tree!(min,max,value::Float32)'
'value=normalize_tree_value(min,max,value::Float32)'
"""
mutable struct MinMaxStats
    min::Float32
    max::Float32
end

function update_tree!(treeminmax::MinMaxStats, value::Float32)::Nothing
    treeminmax.min = treeminmax.min < value ? treeminmax.min : value
    treeminmax.max = treeminmax.max > value ? treeminmax.max : value
    return nothing
end

function normalize_tree_value(treeminmax::MinMaxStats, value::Float32)::Float32
    if treeminmax.max > treeminmax.min
        return (value - treeminmax.min) / (treeminmax.max - treeminmax.min)
    else
        return value
    end
end

"""
Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

Returns:
Positive float.
"""
function visit_softmax_temperature_fn(trained_steps::Int)::Float32
    if trained_steps < 500e3
        return 1.0
    elseif trained_steps < 750e3
        return 0.5
    else
        return 0.25
    end
end

using Distributions:Dirichlet, Categorical
using Parameters:@with_kw
using Flux:softmax

@with_kw mutable struct Node
    visit_count::Int = 0
    to_play::Int = 1
    prior::Float32
    value_sum::Float32 = 0.0
    children::Union{Dict{Int,Node},Nothing} = nothing
    hidden_state::Union{Array{Float32,3},Nothing} = nothing
    reward::Union{Float32,Int} = 0
end

function expanded(node::Node)::Bool
    return !isnothing(node.children)
end

function node_value(node::Node)::Float32
    if node.visit_count == 0
        return 0
    else
        return node.value_sum / node.visit_count
    end
end

"""
We expand a node using the value, reward and policy prediction obtained from the
neural network.
"""
function expand_node!(node::Node, actions, to_play::Int, reward::Float32, policy_logits::Vector{Float32}, hidden_state::Array{Float32,3})::Nothing
    policy_values = softmax([policy_logits[a] for a in actions])
    policy = Dict([(a, policy_values[i]) for (i, a) in enumerate(actions)])
    node.children = Dict([(action, Node(prior=prob)) for (action, prob) in policy])
    node.to_play = to_play
    node.reward = reward
    node.hidden_state = hidden_state
	return nothing
end

"""
At the start of each search, we add dirichlet noise to the prior of the root to
encourage the search to explore new actions.
"""
function add_exploration_noise!(node::Node, dirichlet_α::Float32, exploration_ϵ::Float32)::Nothing
    actions = collect(keys(node.children))
    noise = rand(Dirichlet(length(actions), dirichlet_α))
    for (a, n) in zip(actions, noise)
        node.children[a].prior = node.children[a].prior * (1 - exploration_ϵ) + n * exploration_ϵ
    end
	return nothing
end


"""
Stores MCTS stats like `child_visits` and `root_value` to GameHistory
"""
function store_search_stats!(history::GameHistory, root::Node, action_space::Array{Int})
	children=collect(values(root.children))
	children=filter!(x->!isnothing(x), children)
	sum_visits = sum([child.visit_count for child in children])
	history.child_visits = hcat(history.child_visits, [haskey(root.children, a) ? root.children[a].visit_count / sum_visits : 0.0f0 for a in action_space])
	value=node_value(root)
	append!(history.root_values, value)
end

"""
Generate a new observation with the observation at the index position
and 'num_stacked_observations' past observations and actions stacked.
"""
function get_stacked_observations(history::GameHistory, index::Int, num_stacked_observations::Int)::Array{Float32,3}

    # stacked_observations = Array{Float32}(undef, conf.observation_shape[1:2]...,0)
	stacked_observations = copy(history.observation_history[:,:,:,index])
    for past_observation_index = index-1:-1:index - num_stacked_observations
        if 1 <= past_observation_index
            previous_observation = cat(
				ones(Float32, conf.observation_shape[1:2]) .* history.action_history[past_observation_index],
				history.observation_history[:,:,:,past_observation_index],
				dims=3
				)
		else
            previous_observation = cat(
				zeros(Float32, conf.observation_shape[1:2]),
				zeros(Float32, conf.observation_shape),
				dims=3
				)
			end
        stacked_observations = cat(stacked_observations, previous_observation, dims=3)
    end
    return stacked_observations
end

using Random
global rng = MersenneTwister(1234)

"""
Select the child with the highest UCB score.
"""
function select_child(node::Node, treeminmax::MinMaxStats)::Tuple{Int,Node}
	actions= collect(keys(node.children))
	children= collect(values(node.children))
	ucb_scores=[ucb_score(node, child, treeminmax) for child in children]
    max_ucb = maximum(ucb_scores)
	max_ucbs=findall(x-> x==max_ucb, ucb_scores)
	i = 0
    i = rand(max_ucbs)
	return actions[i], children[i]
end

"""
The score for a node is based on its value, plus an exploration bonus based on the prior.
"""
function ucb_score(parent_node::Node, child::Node, treeminmax::MinMaxStats)::Float32
    pb_c = (log2((parent_node.visit_count + conf.pb_c_base + 1) / conf.pb_c_base)
            + conf.pb_c_init)
    pb_c *= sqrt(parent_node.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0
        # Mean value Q
        value_score = normalize_tree_value(treeminmax, child.reward + (conf.discount * (length(conf.players) == 1 ? node_value(child) : -node_value(child))))
    else
        value_score = 0
    end
    return prior_score + value_score
end

"""
At the end of a simulation, we propagate the evaluation all the way up the tree
to the root.
"""
function backpropagate!(search_path::Vector{Node}, value::Float32, to_play::Int, treeminmax::MinMaxStats)::Nothing
    if length(conf.players) == 1
        for node in reverse(search_path)
            node.value_sum += value
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + conf.discount * node_value(node))
            value = node.reward + conf.discount * value
        end
    elseif length(conf.players) == 2
        for node in reverse(search_path)
            if node.to_play == to_play 
                node.value_sum += value
            else
                node.value_sum -= value
            end
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + conf.discount * node_value(node))
            if node.to_play == to_play
                value = -node.reward
            else
                value = node.reward + conf.discount * value
            end
        end
    else
        ErrorException("backpropagate for more than 2 players is not implemented")
	end
	return nothing
end

"""
# Game Independent Monte Carlo Tree Search algorithm.
To decide on an action, we run N simulations, always starting at the root of
the search tree and traversing the tree according to the UCB formula until we
reach a leaf node.

At the root of the search tree we use the representation function to obtain a
hidden state given the current observation.
We then run a Monte Carlo Tree Search using only action sequences and the network
learned by the network.
"""
function run_mcts(observation::Array{Float32,3}, legal_actions::Vector{Int}, to_play::Int, exploration::Bool, NNs::NamedTuple{(:representation, :prediction, :dynamics), Tuple{Any, Any, Any}})::Node
    
	root = Node(prior=0.)
	observation = unsqueeze(observation, 4)
	hidden_state = NNs.representation(observation)
	if ndims(hidden_state)==2
		hidden_state=reshape(hidden_state, (conf.observation_shape...,1))
	end
	# hidden_state = unsqueeze(hidden_state, 4)
	root_predicted_value, policy_logits = NNs.prediction(hidden_state)
	hidden_state, root_predicted_value, policy_logits = squeeze.([hidden_state, root_predicted_value, policy_logits])
	reward = 0.0f0
	# policy_logits = unsqueeze(policy_logits, 2)
	@assert !isempty(legal_actions) "Legal actions should not be an empty array. Got $(legal_actions)"
	@assert Set(legal_actions) ⊆ Set(conf.action_space) "Legal actions should be a subset of the action space."
	expand_node!(root, legal_actions, to_play, reward, policy_logits, hidden_state)
    
	if exploration
        add_exploration_noise!(root, conf.dirichlet_α, conf.exploration_ϵ)
    end

    treeminmax = MinMaxStats(Inf, -Inf)

    max_tree_depth = 0
    for iter = 1:conf.num_iters
		# @info "Running MCTS: currently at iteration $(iter)"
        node = root
		virtual_to_play = to_play
        search_path = [node]
        current_tree_depth = 0
		action=0
        while expanded(node)
            current_tree_depth += 1
            action, node = select_child(node, treeminmax)
            push!(search_path, node)
			
			# Players play turn by turn
			virtual_to_play = mod1(virtual_to_play + 1, length(conf.players))
        end

        parent = search_path[end-1]
		value, policy_logits = NNs.prediction(unsqueeze(parent.hidden_state, 4))
		# policy_logits = unsqueeze(policy_logits, 2)
		state_action = make_state_action(parent.hidden_state, action, conf)
		state_action = unsqueeze(state_action, 4)
		next_hidden_state, reward = NNs.dynamics(state_action)
		if ndims(next_hidden_state)==2
			next_hidden_state=reshape(next_hidden_state, (conf.observation_shape...,1))
		end
		value, policy_logits, next_hidden_state, reward = squeeze.([value, policy_logits, next_hidden_state, reward])
        expand_node!(node, legal_actions, virtual_to_play, reward[1], policy_logits, next_hidden_state)
        backpropagate!(search_path, value[1], virtual_to_play, treeminmax)
        max_tree_depth = maximum([max_tree_depth, current_tree_depth])
    end
    return root
end


"""
Select action according to the visit count distribution and the temperature.
The temperature is changed dynamically with the visit_softmax_temperature function
in the config.
"""
function select_action(node::Node, temperature::Float32)::Int
    visit_counts = Int32[child.visit_count for child in values(node.children)]
    actions = [action for action in keys(node.children)]
    if temperature == 0.0f0
        action = actions[argmax(visit_counts)]
    elseif temperature == Inf
        action = rand(rng, actions)
    else
        visit_count_distribution = visit_counts.^(1 / temperature)
        visit_count_distribution = visit_count_distribution ./ sum(visit_count_distribution)
        action = actions[rand(rng, Categorical(visit_count_distribution))]
    end
    return action
end

"""
Select opponent action for evaluating MuZero level.
"""
function select_opponent_action(env::AbstractEnv, opponent::String, stacked_observations::Array{Float32,3})::Int
    if opponent == "human"
		p = current_player(env)
		las = legal_action_space(env, p)
        return human_input()
    elseif opponent == "expert"
        return expert_agent()
    elseif opponent == "random"
        @assert las "Legal actions should not be an empty array. Got $(legal_actions)"
        @assert Set(las) ⊆ Set(conf.action_space) "Legal actions should be a subset of the action space."
        return rand(rng, las)
    else
        error("Wrong argument: opponent argument should be self, human, expert or random")
    end
end

"""
Play one game with actions based on the Monte Carlo tree search at each moves.
"""
function play_game(env::AbstractEnv, temperature, render::Bool, opponent::String, muzero_player::Int, NNs::NamedTuple{(:representation, :prediction, :dynamics), Tuple{Any, Any, Any}})::GameHistory
    history = GameHistory(Array{Float32}(undef, conf.observation_shape..., 0), Vector{Int}(), Vector{Float32}(), Vector{Int}(), Matrix{Float32}(undef, length(conf.action_space), 0), Vector{Float32}(), nothing, nothing, nothing)
	
    done = false
	
    if render
        render_game(env)
    end
	
	observation=0
	root=0
	action=0

    while !done && length(history.action_history) <= conf.max_moves
		if !isnothing(conf.temperature_threshold) && length(history.action_history) ≥ conf.temperature_threshold
			temperature = 0.0f0
		end
		
		if length(history.action_history)==0
			observation = reset!(env)
		end
		p = current_player(env)
    	history.observation_history = cat(history.observation_history, observation, dims=4)
		# @info "Self-Playing one game: $(length(history.action_history)-1) moves played"
        @assert ndims(observation) == 3 "Observation should be 3 dimensional instead of $(ndims(observation)) dimensionnal. Got observation of shape: $(size(observation))"
        stacked_observations = get_stacked_observations(history, lastindex(history.observation_history, 4), conf.stacked_observations)

        # Choose the action
        if opponent == "self" || muzero_player == p
            root = run_mcts(stacked_observations, legal_action_space(env, p), p, true, NNs)
            action = select_action(root, temperature)
        else
            action = select_opponent_action(env, opponent, stacked_observations)
        end

        # observation, reward, done = execute_step(action)
        observation = env(action) # execute step and return observation, changes the current player
        reward = convert(Float32, RLBase.reward(env, p))
        done = is_terminated(env) #TODO instead of initializing state info everytime 

        if render
        	println("Played action: $(action)")
            render_game(env)
        end

        store_search_stats!(history, root, conf.action_space)

        append!(history.action_history, action)
        append!(history.reward_history, reward)
        push!(history.to_play_history, p)
    end
    return history
end

function self_play!(env,
training_step,
num_played_games,
num_played_steps,
total_samples,
remote_NNs,
remote_buffer::RemoteChannel{BufferChannel})::Bool

	NNs = fetch(remote_NNs)
	training_step_ = 0
    while training_step_ ≤ conf.training_steps

		training_step_ = take!(training_step)
		temperature = visit_softmax_temperature_fn(training_step_)

		if training_step_ % conf.checkpoint_interval == 0 && training_step_ > 1
			NNs = take!(remote_NNs)
			@info "Networks loaded during SelfPlay" training_step_ #size(fetch(remote_buffer))
		end

		# Explore moves during training mode
		history = play_game(
			env,
			temperature,
			false,
			"self",
			conf.muzero_player,
			NNs
		)
		# @info "One episode of Self-Play finished"
		save_game(history, remote_buffer, num_played_games,
			num_played_steps,
			total_samples,)
    end
	return true
end

function competitive_play!(;buffer_to_disk=false, NNs)::Nothing
# Take the best action (no exploration) in competition mode
history = play_game(
	env,
	0.0f0,
	true,
	length(conf.players) == 1 ? "self" : conf.opponent,
	conf.muzero_player,
	NNs
)
# if buffer_to_disk
	# save_game(history, buffer)
# end
return nothing
end