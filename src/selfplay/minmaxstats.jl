"""
# Holds the Min-Max values of the tree
Simply brings variables('tree_min' and 'tree_max') and functions('update' and 'normalize')
to the scope.
'tree_min,tree_max=MinMaxStats.update(tree_min,tree_max,value::Float64)'
'value=MinMaxStats.normalize(tree_min,tree_max,value::Float64)'
"""

export treeminmax, update!, normalize

mutable struct TreeMinMax
    tree_min::Float64
    tree_max::Float64
end
treeminmax = TreeMinMax(Inf, -Inf)
# tree_min = Inf
# tree_max = -Inf

function update!(treeminmax::TreeMinMax, value::Float64)::Nothing
    treeminmax.tree_min = treeminmax.tree_min < value ? treeminmax.tree_min : value
    treeminmax.tree_max = treeminmax.tree_max > value ? treeminmax.tree_max : value
    return nothing
end

function normalize(treeminmax::TreeMinMax, value::Float64)::Float64
    if treeminmax.tree_max > treeminmax.tree_min
        return (value - treeminmax.tree_min) / (treeminmax.tree_max - treeminmax.tree_min)
    else
        return value
    end
end

# function update(tree_min::Float64,tree_max::Float64,value::Float64)::Tuple{Float64, Float64}
#     min=minimum([tree_min,value])
#     max=maximum([tree_max,value])
#     return min, max
# end

# function normalize(tree_min::Float64,tree_max::Float64,value::Float64)::Float64
#     if tree_max > tree_min
#         return (value - tree_min)/(tree_max - tree_min)
#     else
#         return value
#     end
# end
