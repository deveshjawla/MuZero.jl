"""
Inherit this class for muzero to play
"""
module AbstractGame


"""
Apply action to the game.

Args:
    action : action of the action_space to take.

Returns:
    The new observation, the reward and a boolean if the game has ended.
"""
function step(action)
    nothing
end

"""
Return the current player.

Returns:
    The current player, it should be an element of the players list in the config. 
"""
function to_play(self)
    return 0
end


"""
Should return the legal actions at each turn, if it is not available, it can return
the whole action space. At each turn, the game have to be able to handle one of returned actions.

For complex game where calculating legal moves is too long, the idea is to functionine the legal actions
equal to the action space but to return a negative reward if the action is illegal.

Returns:
    An array of integers, subset of the action space.
"""
function legal_actions(self)
    nothing
end


"""
Reset the game for a new game.

Returns:
    Initial observation of the game.
"""
function reset(self)
    nothing
end

"""
Properly close the game.
"""
function close(self)
    nothing
end


"""
Display the game observation.
"""
function render(self)
    nothing
end

"""
For multiplayer games, ask the user for a legal action
and return the corresponding action number.

Returns:
    An integer from the action space.
"""
function human_to_action(self)
    choice = input("Enter the action to play for the player {to_play()}: ")
    while int(choice) ∉ legal_actions()
        choice = input("Ilegal action. Enter another action : ")
    end
    return int(choice)
end

"""
Hard coded agent that MuZero faces to assess his progress in multiplayer games.
It doesn't influence training

Returns:
    Action as an integer to take in the current game state
"""
function expert_agent(self)
    error("unimplemented")
end

"""
Convert an action number to a string representing the action.

Args:
    action_number: an integer from the action space.

Returns:
    String representing the action.
"""
function action_to_string(action_number)
    return "action_number"
end
end
