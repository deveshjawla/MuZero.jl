export Node, expanded, node_value, expand_node, add_exploration_noise
using Distributions: Dirichlet
using Parameters

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
function expand_node(node::Node, actions, to_play::Int64, reward::Float64, policy_logits, hidden_state::State)::Node
    policy_values = [] #TODO
    policy = Dict([a, policy_values[i] for (i, a) in actions])
    node.children = Dict([
        (action, Node( to_play=-node.to_play, prior=prob)) for
        (action, prob) in policy
    ])
    node.to_play = to_play
    node.reward = reward
    node.hidden_state = hidden_state

    return node
end

"""
At the start of each search, we add dirichlet noise to the prior of the root to
encourage the search to explore new actions.
"""
function add_exploration_noise(node::Node, dirichlet_α::Float64, exploration_ϵ::Float64)::Node
    actions = collect(keys(node.children))
    noise = rand(Dirichlet(length(actions), dirichlet_α))
    node.children = Dict([
        (
            a,
            Node(
                to_play= -node.to_play,
                prior = node.children[a].prior * (1 - exploration_ϵ) + n * exploration_ϵ,
                ),
        ) for (a, n) in zip(actions, noise)
    ])
    return node
end