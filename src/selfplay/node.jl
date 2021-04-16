
using Distributions: Dirichlet
export Node, expanded, value, expand_node, add_exploration_noise

struct Node
    visit_count::Int64
    to_play::Int64
    prior::Float64
    value_sum::Float64
    children::Dict{Int64,Node}
    hidden_state::State
    reward::Int64
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
function expand_node(node::Node, actions, to_play, reward, policy_logits, hidden_state)::Node
    policy_values = [] #TODO
    policy = Dict([a, policy_values[i] for (i, a) in actions])
    children = Dict([
        (action, Node(0, -node.to_play, prob, 0.0, nothing, nothing, 0.0)) for
        (action, prob) in policy
    ])
    return Node(
        node.visit_count,
        to_play,
        node.prior,
        node.value_sum,
        children,
        hidden_state,
        reward,
    )
end

"""
At the start of each search, we add dirichlet noise to the prior of the root to
encourage the search to explore new actions.
"""
function add_exploration_noise(node::Node, dirichlet_α, exploration_ϵ)::Node
    actions = collect(keys(node.children))
    noise = rand(Dirichlet(length(actions), dirichlet_α))
    children = Dict([
        (
            action,
            Node(
                0,
                -node.to_play,
                node.children[a].prior * (1 - exploration_ϵ) + n * exploration_ϵ,
                0.0,
                nothing,
                nothing,
                0.0,
            ),
        ) for (a, n) in zip(actions, noise)
    ])
    return Node(
        node.visit_count,
        to_play,
        node.prior,
        node.value_sum,
        children,
        hidden_state,
        reward,
    )
end
