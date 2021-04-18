module AbstractNetwork


function initial_inference(self, observation) end



function recurrent_inference(self, encoded_state, action) end


function get_weights(self)
    return dict_to_cpu(state_dict())
end


function set_weights(self, weights)
    load_state_dict(weights)
end


end


