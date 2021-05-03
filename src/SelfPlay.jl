"""
Run in a dedicated thread to play games and save them to the replay-buffer.
"""
# export GameHistory, store_search_stats!, get_stacked_observations
# export select_child, ucb_score, backpropagate, run_mcts

include("ReplayBuffer.jl")

"""
# Holds the Min-Max values of the tree
Simply brings variables('min' and 'max') and functions('update' and 'normalize')
to the scope.
'update_tree!(min,max,value::Float64)'
'value=normalize_tree_value(min,max,value::Float64)'
"""
mutable struct MinMaxStats
    min::Float64
    max::Float64
end

function update_tree!(treeminmax::MinMaxStats, value::Float64)::Nothing
    treeminmax.min = treeminmax.min < value ? treeminmax.min : value
    treeminmax.max = treeminmax.max > value ? treeminmax.max : value
    return nothing
end

function normalize_tree_value(treeminmax::MinMaxStats, value::Float64)::Float64
    if treeminmax.max > treeminmax.min
        return (value - treeminmax.min) / (treeminmax.max - treeminmax.min)
    else
        return value
    end
end


using Distributions: Dirichlet
using Parameters
using Flux: softmax

@with_kw mutable struct Node
    visit_count::Int64 = 0
    to_play::Int64 = -1
    prior::Union{Float64,Nothing} = nothing
    value_sum::Float64 = 0.0
    children::Union{Dict{Int64,Node},Nothing} = nothing
    hidden_state::Union{State,nothing} = nothing
    reward::Int64 = 0
end

function expanded(node::Node)::Bool
    return length(node.children) > 0
end

function node_value(node::Node)::Float64
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
function expand_node(node::Node, actions, to_play::Int64, reward::Float64, policy_logits, hidden_state::State)::nothing
    policy_values = softmax([policy_logits[1][a] for a in actions])
    policy = Dict([a, policy_values[i] for (i, a) in enumerate(actions)])
    node.children = Dict([(action, Node(prior=prob)) for (action, prob) in policy])
    node.to_play = to_play #TODO
    node.reward = reward
    node.hidden_state = hidden_state
end

"""
At the start of each search, we add dirichlet noise to the prior of the root to
encourage the search to explore new actions.
"""
function add_exploration_noise(node::Node, dirichlet_α::Float64, exploration_ϵ::Float64)::nothing
    actions = collect(keys(node.children))
    noise = rand(Dirichlet(length(actions), dirichlet_α))
    for (a, n) in zip(actions, noise)
        node.children[a].prior = node.children[a].prior * (1 - exploration_ϵ) + n * exploration_ϵ
    end
end


@with_kw mutable struct GameHistory
    observation_history::Vector{Any} = [] #TODO check types
    action_history::Vector{Int64} = []
    reward_history::Vector{Float64} = []
    to_play_history::Vector{Int64} = []
    child_visits::Vector{Int64} = []
    root_values::Vector{Float64} = []
    reanalysed_predicted_root_values = nothing
    priorities = nothing
    game_priority = nothing
end


function store_search_stats!(history::GameHistory, root::Node, action_space::Array{Int64})::nothing
    if root.prior ≠ nothing
        sum_visits = sum([child.visit_count for child in collect(values(root.children))])
        append!(history.child_visits, [haskey(root.children, a) ? root.children[a].visit_count / sum_visits : 0 for a in action_space])
        append!(history.root_values, node_value(root))
    else
        append!(history.root_values, nothing)
    end
end

"""
Generate a new observation with the observation at the index position
and 'num_stacked_observations' past observations and actions stacked.
"""
function get_stacked_observations(history::GameHistory, index::Int64, num_stacked_observations::Inf64)::Vector{Any}
    # Convert to positive index
    index = mod(index, length(history.observation_history))

    stacked_observations = copy(history.observation_history[index+1])
    for past_observation_index = index:-1:index-num_stacked_observations
        if 0 <= past_observation_index
            previous_observation = vcat(
                history.observation_history[past_observation_index],
                [ ones(eltype(stacked_observations[1]),size(stacked_observations[1])) .* history.action_history[past_observation_index] ],
            )
        else
            previous_observation = vcat(
                zeros(eltype(history.observation_history[index+1]),size(history.observation_history[index+1])),
                zeros(eltype(stacked_observations[1]),size(stacked_observations[1])),
            )
        end
        stacked_observations = vcat(stacked_observations, previous_observation)
    end
    return stacked_observations
end


"""
# Game Independent Monte Carlo Tree Search algorithm.
To decide on an action, we run N simulations, always starting at the root of
the search tree and traversing the tree according to the UCB formula until we
reach a leaf node.
"""

struct MCTSParams
    dirichlet_α::Float64 = 0.25
    exploration_ϵ::Float64 = 0.25
    pb_c_base::Int64 = 19652
    pb_c_init::Float64 = 1.25
    discount::Float64 = 0.997
    num_iters::Int64 = 50
    players::Vector{Int} = collect(1:2) #TODO appears also in SelfPlayParams
end


using Random
rng = MersenneTwister(1234)

"""
Select the child with the highest UCB score.
"""
function select_child(mctsparams::MCTSParams, node::Node, treeminmax::MinMaxStats)::Tuple{Int64,Node}
    max_ucb = maximum([ucb_score(mctsparams,node, child, treeminmax) for (action, child) in node.children])
    action = rand(rng,[ucb_score(mctsparams,node, child, treeminmax) == max_ucb ? action : nothing for (action, child) in node.children])
    return action, node.children[action]
end

"""
The score for a node is based on its value, plus an exploration bonus based on the prior.
"""
function ucb_score(mctsparams::MCTSParams, parent::Node, child::Node, treeminmax::MinMaxStats)::Float64
    pb_c = (log2((parent.visit_count + mctsparams.pb_c_base + 1) / mctsparams.pb_c_base)
            + mctsparams.pb_c_init)
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0
        # Mean value Q
        value_score = normalize_tree_value(treeminmax, child.reward + (mctsparams.discount * (length(mctsparams.players) == 1 ? node_value(child) : -node_value(child))))
    else
        value_score = 0
    end
    return prior_score + value_score
end

"""
At the end of a simulation, we propagate the evaluation all the way up the tree
to the root.
"""
function backpropagate(mctsparams::MCTSParams, search_path::Vector{Node}, value::Float64, to_play::Int64, treeminmax::MinMaxStats)::nothing
    if length(mctsparams.players) == 1
        for node in reverse(search_path)
            node.value_sum += value
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + mctsparams.discount * node_value(node))
            value = node.reward + mctsparams.discount * value
        end
    elseif length(mctsparams.players) == 2
        for node in reverse(search_path)
            if node.to_play == to_play 
                node.value_sum += value
            else
                node.value_sum-=value
            end
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + mctsparams.discount * node_value(node))
            if node.to_play == to_play
                value= -node.reward
            else
                value = node.reward + mctsparams.discount * value
            end
        end
    else
        ErrorException("Env for more than 2 players is not implemented")
    end
end

"""
At the root of the search tree we use the representation function to obtain a
hidden state given the current observation.
We then run a Monte Carlo Tree Search using only action sequences and the network
learned by the network.
"""
function run_mcts(mctsparams::MCTSParams,
    observation::Vector{Observation}, #TODO
    legal_actions::Vector{Int64},
    to_play::Int64,
    exploration_noise::Bool;
    override_root_with=nothing
)::Tuple{Node,Dict{String,Number}}
    if ≠ isnothing(override_root_with)
        root = override_root_with
        root_predicted_value = nothing
    else
        root = Node(prior = 0)
        hidden_state, value, policy_logits, reward = initial_inference(observation)
        root_predicted_value = support_to_scalar() #TODO
        reward = support_to_scalar() #TODO
        @assert legal_actions "Legal actions should not be an empty array. Got $(legal_actions)"
        @assert set(legal_actions) ⊆ set(mctsparams.action_space) "Legal actions should be a subset of the action space."
        expand_node(root, legal_actions, to_play, reward, policy_logits, hidden_state)
    end

    if exploration_noise
        add_exploration_noise(root, mctsparams.dirichlet_α, mctsparams.exploration_ϵ)
    end

    treeminmax = MinMaxStats(Inf, -Inf)

    max_tree_depth = 0
    for _ = 1:mctsparams.num_iters
        virtual_to_play = to_play
        node = root
        search_path = [node]
        current_tree_depth = 0
        while expanded(node)
            current_tree_depth += 1
            action, child = select_child(mctsparams, node, treeminmax)
            append!(search_path, child)

            # Players play turn by turn
            if virtual_to_play + 1 < length(mctsparams.players)
                virtual_to_play = mctsparams.players[virtual_to_play]
            else
                virtual_to_play = mctsparams.players[1]
            end
        end

        # Inside the search tree we use the dynamics function to obtain the next hidden
        # state given an action and the previous hidden state
        parent = search_path[end-1]
        value, reward, policy_logits, next_hidden_state = recurrent_inference(parent.hidden_state, action) #TODO
        value = support_to_scalar() #TODO in Py basically this has to do with converting from torch to floats
        reward = support_to_scalar() #TODO
        expand_node(root, legal_actions, to_play, reward, policy_logits, next_hidden_state)
        backpropagate(mctsparams,search_path, value, virtual_to_play, treeminmax)
        max_tree_depth = maximum(max_tree_depth, current_tree_depth)
    end
    extra_info = Dict(
        "max_tree_depth" => max_tree_depth,
        "root_predicted_value" => root_predicted_value,
    )
    return root, extra_info
end

struct SelfPlayParams
    num_workers::Int64 = 350
    selfplay_on_gpu::Bool = false
    max_moves::Int64 = 27000
    temperature_threshold::Nothing = nothing
    players::Vector{Int} = collect(1:2)
end

"""
Select action according to the visit count distribution and the temperature.
The temperature is changed dynamically with the visit_softmax_temperature function
in the config.
"""
function select_action(node::Node, temperature::Float64)::Int64
    visit_counts = Int32[child.visit_count for child in values(node.children)]
    actions = [action for action in keys(node.children)]
    if temperature == 0
        action = actions[argmax(visit_counts)]
    elseif temperature == Inf
        action = rand(rng, actions)
    else
        visit_count_distribution = visit_counts .^ (1 / temperature)
        visit_count_distribution = visit_count_distribution ./ sum(visit_count_distribution)
        action = actions[rand(rng, Categorical(visit_count_distribution))]
    end
    return action
end

"""
Select opponent action for evaluating MuZero level.
"""
function select_opponent_action(mctsparams::MCTSParams, env::AbstractEnv, opponent::String, stacked_observations::Vector{Observation})
    if opponent == "human"
        root, mcts_info = run_mcts(
            mctsparams,
            stacked_observations,
            RLBase.legal_action_space(env, p),#TODO
            RLBase.current_player(env),
            true
        )
        print("Tree depth: $(mcts_info["max_tree_depth"])")
        print("Root value for player $(RLBase.current_player(env)): $(node_value(root))")
        print(
            "Player $(RLBase.current_player(env)) turn. MuZero suggests $(action_to_string(select_action(root, 0)))", #TODO
        )
        return human_to_action(), root #TODO
    elseif opponent == "expert"
        return expert_agent(), nothing
    elseif opponent == "random"
        @assert RLBase.legal_action_space(env, p) "Legal actions should not be an empty array. Got $(legal_actions)"
        @assert set(RLBase.legal_action_space(env, p)) ⊆ set(RLBase.action_space(env)) "Legal actions should be a subset of the action space."
        return rand(rng, RLBase.legal_action_space(env, p)), nothing
    else
        error("Wrong argument: opponent argument should be self, human, expert or random")
    end
end

"""
Play one game with actions based on the Monte Carlo tree search at each moves.
"""
function play_game(
    selfplayparams::SelfPlayParams,
    mctsparams::MCTSParams,
    config::GeneralParams,
    env::AbstractEnv,
    temperature,
    temperature_threshold,
    render::Bool,
    opponent::String,
    muzero_player::Int64,
)
    history = GameHistory()
    observation = RLBase.reset!(env)
    append!(history.action_history, 0)
    append!(history.observation_history, observation)
    append!(history.reward_history, 0)
    append!(history.to_play_history, RLBase.current_player(env))

    done = false

    if render
        render_game() #TODO
    end

    while !done && history.action_history <= selfplayparams.max_moves
        @assert length(size(observation)) == 3 "Observation should be 3 dimensional instead of $(length(size(observation))) dimensionnal. Got observation of shape: $(size(observation))"
        @assert size(observation) == config.observation_shape "Observation should match the observation_shape defined in MuZeroConfig. Expected $(config.observation_shape) but got $(size(observation))."
        stacked_observations =
            get_stacked_observations(history, -1, config.stacked_observations)

        # Choose the action
        if opponent == "self" || muzero_player == RLBase.current_player(env)
            root, mcts_info = run_mcts(
                mctsparams,
            stacked_observations,
            RLBase.legal_action_space(env, p),#TODO
            RLBase.current_player(env),
            true
            )
            action = select_action( root, !isnothing(temperature_threshold) || length(history.action_history) < temperature_threshold ? temperature : 0) #CHECK the condition
            if render
                println("Tree depth: $(mcts_info["max_tree_depth"])")
                println(
                    "Root value for player $(RLBase.current_player(env)): $(node_value(root))",
                )
            end

        else
            action, root =
                select_opponent_action(mctsparams, env, opponent, stacked_observations)
        end

        # observation, reward, done = execute_step(action)
        observation = env(action) #execute step and return observation
        reward = RLBase.reward(env, RLBase.current_player(env))
        done = RLBase.is_terminated(env)

        if render
            println("Played action: {action_to_string(action)}")
            render_game() #TODO
        end

        store_search_stats!(history, root, RLBase.action_space(env))

        # Next batch
        append!(history.action_history, action)
        append!(history.observation_history, observation)
        append!(history.reward_history, reward)
        append!(history.to_play_history, RLBase.current_player(env))
    end
    return history
end
include("utilities/Utilities.jl")
include("training/Training.jl")
function continuous_self_play(
    selfplayparams::SelfPlayParams,
    mctsparams::MCTSParams,
    config::GeneralParams,
    env::AbstractEnv,
    trainparams::TrainParams,
    checkpoint::Dict,
    competition_mode = false,
)
    while get_info(checkpoint,"training_step") < trainparams.training_steps && get_info(checkpoint,"terminate")
        set_weights(weights) #TODO
        if !competition_mode
            #Explore moves during training mode
            history = play_game(
                selfplayparams,
                mctsparams,
                config,
                env,
                Utils.visit_softmax_temperature_fn(get_info(checkpoint,"training_step")),
                config.temperature_threshold,
                false,
                "self",
                0,
            )
            save_game(rbp, history, checkpoint, true)
        else
            # Take the best action (no exploration) in competition mode
            history = play_game(
                selfplayparams,
                mctsparams,
                config,
                env,
                Utils.visit_softmax_temperature_fn(get_info(checkpoint,"training_step")),
                config.temperature_threshold,
                false,
                length(mctsparams.players) == 1 ? "self" : config.opponent,
                config.muzero_player,
            )

            #Save to shared_storage
            set_info!(checkpoint,
                Dict(
                    "episode_length" => length(history.action_history) - 1,
                    "total_reward" => sum(history.reward_history),
                    "mean_value" =>
                        mean([value for value in history.root_values if value]),
                ),
            )

            if 1 < length(mctsparams.players)
                set_info!(checkpoint,
                    Dict(
                        "muzero_reward" => sum(
                            [reward for (i, reward) in enumerate(history.reward_history) if
                            history.to_play_history[i] == config.muzero_player]
                        ),
                        "opponent_reward" => sum(
                            [reward for (i, reward) in enumerate(history.reward_history) if
                            history.to_play_history[i] != config.muzero_player]
                        ),
                    ),
                )
            end
        end

        # Managing the self-play / training ratio
        if !competition_mode && config.self_play_delay
            sleep(config.self_play_delay)
        end
        if !competition_mode && !isnothing(config.ratio)
            while get_info(checkpoint,"training_step") / get_info(checkpoint,"num_played_steps") < config.ratio && get_info(checkpoint,"training_step") < trainparams.training_steps && !get_info(checkpoint,"terminate")
                sleep(0.5)
            end
        end
    end
    close_game() #TODO
end

end