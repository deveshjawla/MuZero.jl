using Distributed
addprocs(2, exeflags="--project")

@everywhere begin
include("../../src/Constructors.jl")
include("../../src/RemoteBufferChannel.jl")
include("../../src/SelfPlay.jl")
include("../../src/ReplayBuffer.jl")
include("../../src/Learning.jl")
include("game.jl")
include("params.jl")
end

env=TicTacToe()
training_step = RemoteChannel(()->Channel{Int}(1))
num_played_games = RemoteChannel(()->Channel{Int}(1))
num_played_steps = RemoteChannel(()->Channel{Int}(1))
num_reanalysed_games = RemoteChannel(()->Channel{Int}(1))
total_samples = RemoteChannel(()->Channel{Int}(1))
remote_NNs = RemoteChannel(()->Channel{NamedTuple{(:representation, :prediction, :dynamics), Tuple{Any, Any, Any}}}(1))
remote_buffer = RemoteChannel(()->BufferChannel()) #TODO Start saving Buffer to disk at periodic intervals after a certain stage in the training process

put!(remote_NNs, (representation=init_representation(hyper), prediction=init_prediction(hyper), dynamics=init_dynamics(hyper)))
put!(training_step, 0)
put!(num_played_games, 0)
put!(num_played_steps, 0)
put!(num_reanalysed_games, 0)
put!(total_samples, 0)

sp = @spawnat :any self_play!(env,
training_step,
num_played_games,
num_played_steps,
total_samples,
remote_NNs,
remote_buffer)

learn = @spawnat :any learning!(num_played_games,
training_step,
remote_NNs,
remote_buffer) 

# include("./src/MuZero.jl")
# using .MuZero
# include("./games/tictactoe/game.jl")
# ttt=TicTacToe()
# include("./games/tictactoe/params.jl")
# MuZero.self_play(hyper, ttt, MuZero.progress, MuZero.buffer)
