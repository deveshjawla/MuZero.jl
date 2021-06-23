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

lp=LearningProgress(0)
buffer_stats= BufferStats(0,0,0,0)
"""
Initial Buffer, Used in self-play first to save_game then in replay buffer and learning
"""
buffer = Dict{Int, GameHistory}() #TODO Start saving Buffer to disk at periodic intervals after a certain stage in the training process


representation= init_representation(conf, hyper)
prediction= init_prediction(conf, hyper)
dynamics= init_dynamics(conf, hyper)
# @info "Networks initialised"

ttt=TicTacToe()

@async begin
@spawn self_play(conf, representation, prediction, dynamics, ttt, buffer)
@spawn training(conf, representation, prediction, dynamics, buffer) 
end



# include("./src/MuZero.jl")
# using .MuZero
# include("./games/tictactoe/game.jl")
# ttt=TicTacToe()
# include("./games/tictactoe/params.jl")
# MuZero.self_play(conf, hyper, ttt, MuZero.progress, MuZero.buffer)
