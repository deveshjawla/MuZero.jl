"""
Class which run in a dedicated thread to store the network weights and some information.
"""

module SharedStorage

    export save_network, get_info, set_info!

    function save_network(config,current_checkpoint::Dict,path=nothing)
        if !path
            path = config.results_path * "model.checkpoint"
        end
        save(current_checkpoint, path) #TODO
    end

    function get_info(current_checkpoint,key::String)
        return current_checkpoint[key]
    end

    function get_info(current_checkpoint,keys::Array)
        return Dict(key=> current_checkpoint[key] for key in keys)
    end

    function set_info!(current_checkpoint, key::String, value=nothing)
        if !value
            merge!(current_checkpoint, Dict(key => value))
        end
        return current_checkpoint
    end

    function set_info!(current_checkpoint, keys::Dict)
        merge!(current_checkpoint,keys)
        return current_checkpoint
    end
    
end