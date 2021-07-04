include("../../src/Constructors.jl")
include("../../src/SelfPlay.jl")
include("../../src/ReplayBuffer.jl")
include("../../src/Learning.jl")
include("game.jl")
include("params.jl")

buffer = Dict{Int, GameHistory}()

latest_training_step=1000 #Change this to the iteration number of latest saved networks

# representation= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_representation.bin"))
# prediction= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_prediction.bin"))
# dynamics= deserialize(joinpath(conf.networks_path,"$(latest_training_step)_dynamics.bin"))

representation= init_representation(hyper)
prediction= init_prediction(hyper)
dynamics= init_dynamics(hyper)

NNs = (representation=representation, prediction=prediction, dynamics=dynamics)



env=TicTacToe()

competitive_play!(buffer_to_disk=false)