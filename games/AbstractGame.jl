
"""
Inherit this class for muzero to play
"""
struct Game
    board::Array <: Any
    player::Int
end


"""
Apply action to the game.

Args:
    action : action of the action_space to take.

Returns:
    The new observation, the reward and a boolean if the game has ended.
"""
function execute_step(action)
    nothing
end

"""
Return the current player.

Returns:
    The current player, it should be an element of the players list in the config. 
"""
function to_play()
    return 0
end


"""
Should return the legal actions at each turn, if it is not available, it can return
the whole action space. At each turn, the game have to be able to handle one of returned actions.

For complex game where calculating legal moves is too long, the idea is to functionize the legal actions
equal to the action space but to return a negative reward if the action is illegal.

Returns:
    An array of integers, subset of the action space.
"""
function legal_actions()
    nothing
end


"""
Reset the game for a new game.

Returns:
    Initial observation of the game.
"""
function reset_game()
    nothing
end

"""
Properly close the game.
"""
function close_game()
    nothing
end



"""
For multiplayer games, ask the user for a legal action
and return the corresponding action number.

Returns:
An integer from the action space.
"""
function human_to_action()
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
function expert_agent()
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

# Extras

function get_observation(game::Game)
    return observation
end

function have_winner()::Bool end

function expert_action()
    
end

"""
Display the game observation.
"""
function render_game(game::Game)
    println(game.board)
end
