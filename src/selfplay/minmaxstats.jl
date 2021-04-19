"""
# Holds the Min-Max values of the tree
Simply brings variables('min' and 'max') and functions('update' and 'normalize')
to the scope.
'min,max=MinMaxStats.update(min,max,value::Float64)'
'value=MinMaxStats.normalize(min,max,value::Float64)'
"""

export MinMaxStats, update_tree!, normalize_tree_value

mutable struct MinMaxStats
    min::Float64
    max::Float64
end

function update_tree!(treeminmax::MinMaxStats, value::Float64)::Nothing
    treeminmax.min = treeminmax.min < value ? treeminmax.min : value
    treeminmax.max = treeminmax.max > value ? treeminmax.max : value
    return nothing
end

function normalize_tree_value(treeminmax::MinMaxStats, value::Float64)::Float64
    if treeminmax.max > treeminmax.min
        return (value - treeminmax.min) / (treeminmax.max - treeminmax.min)
    else
        return value
    end
end
