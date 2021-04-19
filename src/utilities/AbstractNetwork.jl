

function initial_inference( observation) end



function recurrent_inference( encoded_state, action) end


function get_weights()
    return dict_to_cpu(state_dict())
end


function set_weights( weights)
    load_state_dict(weights)
end


