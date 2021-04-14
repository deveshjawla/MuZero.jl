######
###### Holds the Min-Max values of the tree
######

module MinMaxStats
    export tree_min,tree_max,update,normalize

    """
    Simply brings variables('tree_min' and 'tree_max') and functions('update' and 'normalize')
    to the scope.
    'tree_min,tree_max=MinMaxStats.update(tree_min,tree_max,value::Float64)'
    'value=MinMaxStats.normalize(tree_min,tree_max,value::Float64)'
    """

    tree_min=Inf
    tree_max=-Inf

    function update(tree_min::Float64,tree_max::Float64,value::Float64)::Tuple{Float64, Float64}
        min=minimum([tree_min,value])
        max=maximum([tree_max,value])
        return min, max
    end
    
    function normalize(tree_min::Float64,tree_max::Float64,value::Float64)::Float64
        if tree_max > tree_min
            return (value - tree_min)/(tree_max - tree_min)
        else
            return value
        end
    end

end