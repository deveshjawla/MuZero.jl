include("utilities/minmaxstats.jl")
# using Main.MinMaxStats # without this it requires doing: Main.MinMaxStats.update() or Main.MinMaxStats.tree_max
MinMaxStats.treeminmax
MinMaxStats.update!(MinMaxStats.treeminmax,0.4)
MinMaxStats.update!(MinMaxStats.treeminmax,0.8)

MinMaxStats.treeminmax

MinMaxStats.normalize(MinMaxStats.treeminmax,.5)
