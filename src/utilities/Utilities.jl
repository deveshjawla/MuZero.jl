
struct GeneralParams
    seed::Int64 = 0
    max_num_gpus::Nothing = nothing

    observation_shape::Tuple{Int64,Int64,Int64} = (20, 20, 3)
    action_space::String = "list(range(4))"
    stacked_observations::Int64 = 32

    muzero_player::Int64 = 0
    opponent::String

    network::String = "resnet"
    support_size::Int64 = 300

    self_play_delay::Int64 = 0
    training_delay::Int64 = 0
    ratio::Nothing = nothing
 end
 
"""
Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

Returns:
    Positive float.
"""
function visit_softmax_temperature_fn(trained_steps::Int64)
    if trained_steps < 500e3
        return 1.0
    elseif trained_steps < 750e3
        return 0.5
    else
        return 0.25
    end
end
