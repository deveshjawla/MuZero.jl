function dict_to_cpu(Dict())
    return Dict()
end

function scalar_to_support(x, support_size)
    return []
end

function support_to_scalar(logits, support_size)
    return nothing
end

function mlp()
    return nothing
end

function conv3x3()
    return nothing
end

struct FullyConnected
action_space_size
full_support_size
representation_network
dynamics_encoded_state_network
dynamics_reward_network
prediction_policy_network
prediction_value_network
end

struct ResNet
action_space_size
full_support_size
block_output_size_reward
block_output_size_value
block_output_size_policy
representation_network
dynamics_network
prediction_network
end