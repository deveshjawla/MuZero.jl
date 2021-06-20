include("../../src/Constructors.jl")
include("../../src/SelfPlay.jl")
include("../../src/ReplayBuffer.jl")
include("../../src/Learning.jl")
include("game.jl")
include("params.jl")

progress = Dict{String, Int}(
"latest_training_step" => 0,
"num_played_games" => 0,
"num_played_steps" => 0,
"num_reanalysed_games" => 0,
"total_samples" => 0
)

buffer = Dict{Int, GameHistory}()

latest_training_step=230 #Change this to the iteration number of latest saved networks

representation= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_representation.bin"))
prediction= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_prediction.bin"))
dynamics= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_dynamics.bin"))

ttt=TicTacToe()

competitive_play(conf, representation, prediction, dynamics, ttt, progress, buffer)