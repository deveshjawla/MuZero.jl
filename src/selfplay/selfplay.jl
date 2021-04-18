"""
Run in a dedicated thread to play games and save them to the replay-buffer.
"""
module SelfPlay

include("mcts.jl")

function __init__(initial_checkpoint, Game, config, seed)
    global config = config
    global game = Game(seed)

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
        if opponent == "self" || muzero_player == self.game.to_play()
            root, mcts_info = run_mcts(
                config,
                model,
                stacked_observations,
                game.legal_actions(),
                game.to_play(),
                true,
            )
            action = select_action(
                root,
                !temperature_threshold ||
                length(history.action_history) < temperature_threshold ? temperature :
                0,
            )
            if render
                println("Tree depth: $(mcts_info["max_tree_depth"])")
                println(
                    "Root value for player $(self.game.to_play()): {root.value():.2f}", #TODO
                )
            end

        else
            action, root =
                select_opponent_action(model, game, opponent, stacked_observations)
        end

        observation, reward, done = self.game.step(action)

        if render
            println("Played action: {self.game.action_to_string(action)}")
            self.game.render() #TODO
        end

        store_search_stats!(history, root, config.action_space)

        # Next batch
        append!(history.action_history, action)
        append!(history.observation_history, observation)
        append!(history.reward_history, reward)
        append!(history.to_play_history, self.game.to_play())
    end
    return history
end

function continuous_self_play(
    config,
    model,
    shared_storage,
    replay_buffer,
    test_mode = False,
)
    #while training_step @ remote processes < config.training_Steps && !terminated @remote
    model.set_weights(weights) #TODO
    if !test_mode
        #Explore moves during training mode
        history = play_game(
            config,
            game,
            config.visit_softmax_temperature_fn(training_steps),
            config.temperature_threshold,
            False,
            "self",
            0,
        )
        replay_buffer.save_game(game_history, shared_storage) #TODO
    else
        # Take the best action (no exploration) in test mode
        history = play_game(
            config,
            game,
            0,
            config.temperature_threshold,
            False,
            length(config.players) == 1 ? "self" : config.opponent,
            config.muzero_player,
        )

        #Save to shared_storage
        shared_storage.set_info(
            Dict(
                "episode_length" => length(history.action_history) - 1,
                "total_reward" => sum(history.reward_history),
                "mean_value" =>
                    mean([value for value in game_history.root_values if value]),
            ),
        ) #TODO

        if 1 < length(config.players)
            shared_storage.set_info(
                Dict(
                    "muzero_reward" => sum(
                        reward for (i, reward) in enumerate(history.reward_history) if
                        history.to_play_history[i-1] == config.muzero_player
                    ),
                    "opponent_reward" => sum(
                        reward for (i, reward) in enumerate(history.reward_history) if
                        history.to_play_history[i-1] != config.muzero_player
                    ),
                ),
            ) #TODO
        end
    end

    # Managing the self-play / training ratio
    if !test_mode && config.self_play_delay #TODO
        sleep(config.self_play_delay)
    end
    if !test_mode && config.ratio #TODO
        while shared_storage.get_info(training_steps) /
              shared_storage.get_info(num_played_steps) < config.ratio &&
                  shared_storage.get_info.remote("training_step") <
                  self.config.training_steps &&
                  !shared_storage.get_info.remote("terminate")
            sleep(0.5)
        end
    end
    close_game()
end

end