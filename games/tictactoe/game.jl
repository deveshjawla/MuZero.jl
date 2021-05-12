using ReinforcementLearningBase

struct Black end
const BLACK = Black()
struct White end
const WHITE = White()

Base.:!(::Black) = WHITE
Base.:!(::White) = BLACK

"""
This is a typical two player, zero sum game. Here we'll also demonstrate how to
implement an environment with multiple state representations.
You might be interested in this [blog](http://www.occasionalenthusiast.com/tag/tic-tac-toe/)
"""
mutable struct TicTacToe <: AbstractEnv
    board::BitArray{3}
    player::Union{Black,White}
end

function TicTacToe()
    board = BitArray{3}(undef, 3, 3, 3) #TODO maybe use StaticArrays?
    fill!(board, false)
    board[:, :, 1] .= true
    TicTacToe(board, WHITE)
end

function reset!(env::TicTacToe)
    fill!(env.board, false)
    env.board[:, :, 1] .= true
    env.player = WHITE
    return env.board
end

const STATE_INFO = Dict{
    TicTacToe,
    NamedTuple{
        (:index, :is_terminated, :winner),
        Tuple{Int,Bool,Union{Nothing,Black,White}},
    },
}()

Base.hash(env::TicTacToe, h::UInt) = hash(env.board, h)
Base.isequal(a::TicTacToe, b::TicTacToe) = isequal(a.board, b.board)

Base.to_index(::TicTacToe, ::White) = 2
Base.to_index(::TicTacToe, ::Black) = 3

action_space(::TicTacToe) = Base.OneTo(9)

legal_action_space(env::TicTacToe, p) = findall(legal_action_space_mask(env, p))

function legal_action_space_mask(env::TicTacToe, p)
    if is_win(env, WHITE) || is_win(env, BLACK)
        zeros(false, 9)
    else
        vec(view(env.board, :, :, 1))
    end
end

(env::TicTacToe)(action::Int) = env(CartesianIndices((3, 3))[action])

function (env::TicTacToe)(action::CartesianIndex{2})
    env.board[action, 1] = false
    env.board[action, Base.to_index(env, env.player)] = true
    env.player = !env.player
    return env.board
end

current_player(env::TicTacToe) = env.player
players(env::TicTacToe) = (WHITE, BLACK)

state(env::TicTacToe, ::Observation{BitArray{3}}, p) = env.board
state_space(env::TicTacToe, ::Observation{BitArray{3}}, p) =
    Space(fill(false..true, 3, 3, 3))

state(env::TicTacToe, ::Observation{Int}, p) =
    get_state_info()[env].index
state_space(env::TicTacToe, ::Observation{Int}, p) =
    Base.OneTo(length(get_state_info()))

function state(env::TicTacToe, ::Observation{String}, p)
    buff = IOBuffer()
    for i in 1:3
        for j in 1:3
            if env.board[i, j, 1]
                x = '.'
            elseif env.board[i, j, 2]
                x = 'x'
            else
                x = 'o'
            end
            print(buff, x)
        end
        print(buff, '\n')
    end
    String(take!(buff))
end
state_space(env::TicTacToe, ::Observation{String}, p) = WorldSpace{String}()


is_terminated(env::TicTacToe) = get_state_info()[env].is_terminated

function reward(env::TicTacToe, player)
    if is_terminated(env)
        winner = get_state_info()[env].winner
        if isnothing(winner)
            0
        elseif winner === player
            1
        else
            -1
        end
    else
        0
    end
end

function is_win(env::TicTacToe, player)
    b = env.board
    p = Base.to_index(env, player)
    @inbounds begin
        b[1, 1, p] & b[1, 2, p] & b[1, 3, p] ||
            b[2, 1, p] & b[2, 2, p] & b[2, 3, p] ||
            b[3, 1, p] & b[3, 2, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 1, p] & b[3, 1, p] ||
            b[1, 2, p] & b[2, 2, p] & b[3, 2, p] ||
            b[1, 3, p] & b[2, 3, p] & b[3, 3, p] ||
            b[1, 1, p] & b[2, 2, p] & b[3, 3, p] ||
            b[1, 3, p] & b[2, 2, p] & b[3, 1, p]
    end
end

function get_state_info()
    if isempty(STATE_INFO)
        @info "initializing state info..."
        t = @elapsed begin
            n = 1
            root = TicTacToe()
            STATE_INFO[root] =
                (index = n, is_terminated = false, winner = nothing)
            walk(root) do env
                if !haskey(STATE_INFO, env)
                    n += 1
                    has_empty_pos = any(view(env.board, :, :, 1))
                    w = if is_win(env, WHITE)
                        WHITE
                    elseif is_win(env, BLACK)
                        BLACK
                    else
                        nothing
                    end
                    STATE_INFO[env] = (
                        index = n,
                        is_terminated = !(has_empty_pos && isnothing(w)),
                        winner = w,
                    )
                end
            end
        end
        @info "finished initializing state info in $t seconds"
    end
    STATE_INFO
end

NumAgentStyle(::TicTacToe) = MultiAgent(2)
DynamicStyle(::TicTacToe) = SEQUENTIAL
ActionStyle(::TicTacToe) = FULL_ACTION_SET
InformationStyle(::TicTacToe) = PERFECT_INFORMATION
StateStyle(::TicTacToe) = (Observation{String}(), Observation{Int}(), Observation{BitArray{3}}())
RewardStyle(::TicTacToe) = TERMINAL_REWARD
UtilityStyle(::TicTacToe) = ZERO_SUM
ChanceStyle(::TicTacToe) = DETERMINISTIC