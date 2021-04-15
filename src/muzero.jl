include("utilities/minmaxstats.jl")
# using Main.MinMaxStats # without this it requires doing: Main.MinMaxStats.update() or Main.MinMaxStats.tree_max

# We won't need to call its methods that many times, but yes for some others using works better
MinMaxStats.treeminmax
MinMaxStats.update!(MinMaxStats.treeminmax,0.4)
MinMaxStats.update!(MinMaxStats.treeminmax,0.8)

MinMaxStats.treeminmax

MinMaxStats.normalize(MinMaxStats.treeminmax,.5)
