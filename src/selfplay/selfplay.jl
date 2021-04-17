"""
Run in a dedicated thread to play games and save them to the replay-buffer.
"""
module SelfPlay

include("mcts.jl")

function __init__(initial_checkpoint, Game, config, seed)
    global config = config
    global game = Game(seed)

    # Fix random generator seed
    torch.manual_seed(seed) #TODO

    # Initialize the network
    global model = models.MuZeroNetwork(config)
    model.set_weights(initial_checkpoint["weights"])
    model.to(torch.device(torch.cuda.is_available() ? "cuda" : "cpu"))
    model.eval()
end

"""
Select action according to the visit count distribution and the temperature.
The temperature is changed dynamically with the visit_softmax_temperature function
in the config.
"""
function select_action(node, temperature)
    visit_counts = Int32[child.visit_count for child in values(node.children)]
    actions = [action for action in keys(node.children)]
    if temperature == 0
        action = actions[argmax(visit_counts)]
    elseif temperature == Inf
        action = rand(rng, actions)
    else
        visit_count_distribution = visit_counts .^ (1 / temperature)
        visit_count_distribution = visit_count_distribution ./ sum(visit_count_distribution)
        action = actions[rand(rng, Categorical(visit_count_distribution))]
    end
    return action
end

"""
Select opponent action for evaluating MuZero level.
"""
function select_opponent_action(model, game, opponent, stacked_observations)
    if opponent == "human"
        root, mcts_info = run_mcts(
            config,
            model,
            stacked_observations,
            game.legal_actions(),
            game.to_play(),
            true,
        )
        print("Tree depth: $(mcts_info["max_tree_depth"])")
        print("Root value for player $(game.to_play()): $(node_value(root))")
        print(
            "Player $(game.to_play()) turn. MuZero suggests $(game.action_to_string(select_action(root, 0)))",
        )
        return game.human_to_action(), root
    elseif opponent == "expert"
        return game.expert_agent(), None
    elseif opponent == "random"
        @assert legal_actions "Legal actions should not be an empty array. Got $(legal_actions)"
        @assert set(legal_actions) ⊆ set(config.action_space) "Legal actions should be a subset of the action space."
        return rand(rng, game.legal_actions()), None
    else
        error("Wrong argument: opponent argument should be self, human, expert or random")
    end
end

"""
Play one game with actions based on the Monte Carlo tree search at each moves.
"""
function play_game(
    config,
    game,
    temperature,
    temperature_threshold,
    render,
    opponent,
    muzero_player,
)
    history = GameHistory()
    observation = self.game.reset()
    append!(history.action_history, 0)
    append!(history.observation_history, observation)
    append!(history.reward_history, 0)
    append!(history.to_play_history, self.game.to_play())

    done = false

    if render
        game.render()
    end

    while !done && history.action_history <= config.max_moves
        @assert length(size(observation)) == 3 "Observation should be 3 dimensional instead of $(length(size(observation))) dimensionnal. Got observation of shape: $(size(observation))"
        @assert size(observation) == config.observation_shape "Observation should match the observation_shape defined in MuZeroConfig. Expected $(config.observation_shape) but got $(size(observation))."
        stacked_observations =
            get_stacked_observations(history, -1, config.stacked_observations)

        # Choose the action
        
    end
end

end