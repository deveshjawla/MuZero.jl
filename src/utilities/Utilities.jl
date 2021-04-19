module Utilities
export visit_softmax_temperature_fn
"""
Parameter to alter the visit count distribution to ensure that the action selection becomes greedier as training progresses.
The smaller it is, the more likely the best action (ie with the highest visit count) is chosen.

Returns:
    Positive float.
"""
function visit_softmax_temperature_fn(config.training_steps,trained_steps::Int64)
    if trained_steps < 500e3
        return 1.0
    elseif trained_steps < 750e3
        return 0.5
    else
        return 0.25

    end
end


end