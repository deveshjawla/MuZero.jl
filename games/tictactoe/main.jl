using Base.Threads:@spawn, @sync, nthreads, @threads, @async
using Distributed: @everywhere
println(nthreads())

@everywhere begin
include("../../src/Constructors.jl")
include("../../src/SelfPlay.jl")
include("../../src/ReplayBuffer.jl")
include("../../src/Learning.jl")
include("game.jl")
include("params.jl")
end


#this should be directly logged with TBL
progress_stats=Dict{String, Float32}(    
	"episode_length" => 0,
    "total_reward" => 0,
    "muzero_reward" => 0,
    "opponent_reward" => 0,
    "mean_value" => 0,
    "total_loss" => 0,
    "value_loss" => 0,
    "reward_loss" => 0,
	)
	
	#TODO instead of these dicts, make a log file
progress = Dict{String, Int}(
"training_step" => 0,
"num_played_games" => 0,
"num_played_steps" => 0,
"num_reanalysed_games" => 0,
"total_samples" => 0
)

buffer = Dict{Int, GameHistory}()

representation= init_representation(conf, hyper)
prediction= init_prediction(conf, hyper)
dynamics= init_dynamics(conf, hyper)
# @info "Networks initialised"

ttt=TicTacToe()

@sync begin
@spawn self_play(conf, hyper, representation, prediction, dynamics, ttt, progress, buffer)
@spawn training(conf, representation, prediction, dynamics, progress, buffer) 
end
# include("./src/MuZero.jl")
# using .MuZero
# include("./games/tictactoe/game.jl")
# ttt=TicTacToe()
# include("./games/tictactoe/params.jl")
# MuZero.self_play(conf, hyper, ttt, MuZero.progress, MuZero.buffer)
