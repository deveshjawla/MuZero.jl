
"""
# GameHistory(ActionHistory)
Keeps track of the action and other relevant GameHistory of a self-play game.
"""
export GameHistory, store_search_stats!, get_stacked_observations
include("node.jl")

@with_kw mutable struct GameHistory
    observation_history::Vector{Any} = [] #TODO sepcify types
    action_history = []
    reward_history = []
    to_play_history = []
    child_visits = []
    root_values = []
    reanalysed_predicted_root_values = nothing
    priorities = nothing
    game_priority = nothing
end

"""
Turn visit_count from root into a policy
"""
function store_search_stats!(history::GameHistory, root::Node, action_space)
    if root.prior ≠ nothing #CHECK
        sum_visits = sum([child.visit_count for child in collect(values(root.children))])
        child_visits = [
            haskey(a, root.children) ? root.children[a].visit_count / sum_visits : 0 for
            a in action_space
        ]
        root_values = node_value(root)
        append!(history.root_values, node_value(root))
    else
        append!(history.root_values, nothing)
    end
end

"""
Generate a new observation with the observation at the index position
and 'num_stacked_observations' past observations and actions stacked.
"""
function get_stacked_observations(history::GameHistory, index, num_stacked_observations)
    # Convert to positive index
    index = mod(index, length(history.observation_history))

    stacked_observations = copy(history.observation_history[index])
    for past_observation_index = index-1:-1:index-num_stacked_observations
        if 0 <= past_observation_index
            previous_observation = vcat(
                history.observation_history[past_observation_index],
                [
                    ones(stacked_observations[1]) *
                    history.action_history[past_observation_index+1],
                ],
            )
        else
            previous_observation = vcat(
                zeros(history.observation_history[index]),
                zeros(stacked_observations[1]), #TOSEE
            )
        end
        stacked_observations = vcat(stacked_observations, previous_observation)
    end
    return stacked_observations
end