"""
# Game Independent Monte Carlo Tree Search algorithm.
To decide on an action, we run N simulations, always starting at the root of
the search tree and traversing the tree according to the UCB formula until we
reach a leaf node.
"""
export select_child, ucb_score, backpropagate, run_mcts

using Random
rng = MersenneTwister(1234)

include("gamehistory.jl")
include("minmaxstats.jl")

include("../utilities/AbstractConfig.jl") #COMMENT this line when running code

"""
Select the child with the highest UCB score.
"""
function select_child(config::Config, node::Node, treeminmax::MinMaxStats)::Tuple{Int64,Node}
    max_ucb = maximum([ucb_score(config,node, child, treeminmax) for (action, child) in node.children])
    action = rand(rng,[ucb_score(config,node, child, treeminmax) == max_ucb ? action : nothing for (action, child) in node.children])
    return action, node.children[action]
end

"""
The score for a node is based on its value, plus an exploration bonus based on the prior.
"""
function ucb_score(config::Config, parent::Node, child::Node, treeminmax::MinMaxStats)::Float64
    pb_c = (log((parent.visit_count + config.pb_c_base + 1) /config.pb_c_base)
            + config.pb_c_init)
    pb_c *= sqrt(parent.visit_count) / (child.visit_count + 1)
    prior_score = pb_c * child.prior

    if child.visit_count > 0
        # Mean value Q
        value_score = normalize_tree_value(treeminmax, child.reward + (config.discount * (length(config.players) == 1 ? node_value(child) : -node_value(child))))
    else
        value_score = 0
    end
    return prior_score + value_score
end

"""
At the end of a simulation, we propagate the evaluation all the way up the tree
to the root.
"""
function backpropagate(config::Config, search_path::Vector{Int64}, value::Float64, to_play::Int64, treeminmax::MinMaxStats)::nothing
    if length(config.players) == 1
        for node in reverse(search_path)
            node.value_sum += value
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + config.discount * node_value(node))
            value = node.reward + config.discount * value
        end
    elseif length(config.players) == 2
        for node in reverse(search_path)
            if node.to_play == to_play 
                node.value_sum += value
            else
                node.value_sum-=value
            end
            node.visit_count += 1
            update_tree!(treeminmax, node.reward + config.discount * node_value(node))
            if node.to_play == to_play
                value= -node.reward
            else
                value = node.reward + config.discount * value
            end
        end
    else
        error("unimplemented")
    end
end

"""
At the root of the search tree we use the representation function to obtain a
hidden state given the current observation.
We then run a Monte Carlo Tree Search using only action sequences and the network
learned by the network.
"""
function run_mcts(config::Config,
    network,
    observations::Vector{Any},
    legal_actions::Vector{Int64},
    to_play::Int64,
    add_exploration_noise::Bool,
    override_root_with::Any,
)::Tuple{Node,Dict{String,Any}}
    if override_root_with
        root = override_root_with
        root_predicted_value = nothing
    else
        root = Node(prior = 0)
        observation = [] #TODO
        root_predicted_value, reward, policy_logits, hidden_state =
            network.initial_inference() #TODO
        root_predicted_value = networks.support_to_scalar() #TODO
        reward = networks.support_to_scalar() #TODO
        @assert legal_actions "Legal actions should not be an empty array. Got $(legal_actions)"
        @assert set(legal_actions) ⊆ set(config.action_space) "Legal actions should be a subset of the action space."
        expand_node(root, legal_actions, to_play, reward, policy_logits, hidden_state)
    end

    if add_exploration_noise
        add_exploration_noise(root, config.dirichlet_α, config.exploration_ϵ)
    end

    treeminmax = MinMaxStats(Inf, -Inf)

    max_tree_depth = 0
    for _ = 1:config.num_simulations
        virtual_to_play = to_play
        node = root
        search_path = [node]
        current_tree_depth = 0
        while expanded(node)
            current_tree_depth += 1
            action, node = select_child(config, node, treeminmax) #TODO
            append!(search_path, node)

            # Players play turn by turn
            if virtual_to_play + 1 < length(config.players)
                virtual_to_play = config.players[virtual_to_play+1]
            else
                virtual_to_play = config.players[1]
            end
        end
        # Inside the search tree we use the dynamics function to obtain the next hidden
        # state given an action and the previous hidden state
        parent = search_path[-2]
        value, reward, policy_logits, hidden_state = network.recurrent_inference() #TODO
        value = networks.support_to_scalar() #TODO
        reward = networks.support_to_scalar() #TODO
        expand_node(root, legal_actions, to_play, reward, policy_logits, hidden_state)
        backpropagate(config,search_path, value, virtual_to_play, treeminmax)
        max_tree_depth = maximum(max_tree_depth, current_tree_depth)
    end
    extra_info = Dict(
        "max_tree_depth" => max_tree_depth,
        "root_predicted_value" => root_predicted_value,
    )
    return root, extra_info
    end
end