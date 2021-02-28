module ContractionOrder
    using LightGraphs
    using LightGraphs: SimpleEdge

    function log2sumexp2(s)
        ms = maximum(s)
        return log2(sum(x->exp2(x - ms), s)) + ms
    end

    function indexof(x, v)
        @inbounds for i=1:length(v)
            if v[i] == x
                return i
            end
        end
        return 0
    end

    include("edgeinfo.jl")
    include("greedy.jl")
end

using .ContractionOrder: order_greedy, TensorNetworkLayout
import .ContractionOrder: abstract_contract, log2sumexp2
using LightGraphs: SimpleEdge, src, dst

export trees_greedy, abstract_contract, log2sumexp2

function build_trees(N::Int, order::AbstractVector)
    ids = collect(Any, 1:N)
    for i=1:length(order)
        ta, tb = src(order[i]), dst(order[i])
        ids[ta] = ContractionTree(ids[ta], ids[tb])
        ids[tb] = nothing
    end
    filter(x->x!==(nothing), ids)
end

function build_order(trees::AbstractVector)
    res = SimpleEdge[]
    for t in trees
        build_order!(t, res)
    end
    return res
end

function build_order!(tree, res)
    if tree isa Integer
        return tree
    else
        a = build_order!(tree.left, res)
        b = build_order!(tree.right, res)
        push!(res, SimpleEdge(a, b))
        return min(a, b)
    end
end

function layout(tn::TensorNetwork)
    graph = tn2graph(tn)
    log2shapes = [[log2.(size(tn.tensors[i]))...] for i=1:length(tn)]
    TensorNetworkLayout(graph, log2shapes)
end

"""
    trees_greedy(tn::TensorNetwork, strategy="min_dim")

Returns a tuple of `(time complexities, space complexities, trees)`,
where `trees` is a vector of `ContractionTree` objects.
For disconnected graphs, the number of trees can be greater than 1.

`strategy` can be "min_dim", "min_reduce", "max_reduce", "min_reduce_tri" or "max_reduce_tri".
"""
function trees_greedy(tn::TensorNetwork; kwargs...)
    tc, sc, order = order_greedy(layout(tn); kwargs...)
    tc, sc, build_trees(length(tn), order)
end

abstract_contract(tn::TensorNetwork, trees::ContractionTree) = abstract_contract(tn, [trees])

function abstract_contract(tn::TensorNetwork, trees::AbstractVector)
    abstract_contract(layout(tn), build_order(trees))
end