export TensorNetworkLayout, order_greedy
export abstract_contract

struct TensorNetworkLayout
    graph::SimpleGraph{Int}
    log2_shapes::Vector{Vector{Float64}}
    function TensorNetworkLayout(graph, log2_shapes)
        @assert nv(graph) == length(log2_shapes) "size of graph ($(nv(graph))) and length of shapes ($(length(log2_shapes))) does not match!"
        new(graph, log2_shapes)
    end
end

LightGraphs.edges(tn::TensorNetworkLayout) = edges(tn.graph)

"""
abstract contraction of  each edge, returns the time complexity and space complexity.
"""
function abstract_contract_edge!(edge, log2_shapes::Vector{Vector{Float64}}, edge_info::EdgeInfo,
        neighbors::Vector{Vector{ET}}, strategy::String) where ET
    haskey(edge_info, edge) || error("edge $edge not found!")
    i, j = src(edge), dst(edge)
    remove!(edge_info, [i, j, neighbors[i]..., neighbors[j]...])
    local log2_shapei, log2_shapej, log2_shapeij
    idxi_j = indexof(j, neighbors[i])
    idxj_i = indexof(i, neighbors[j])
    log2_shapeij = log2_shapes[i][idxi_j]
    deleteat!(log2_shapes[i], idxi_j)
    deleteat!(log2_shapes[j], idxj_i)
    deleteat!(neighbors[i], idxi_j)
    deleteat!(neighbors[j], idxj_i)
    log2_shapei, log2_shapej = sum(log2_shapes[i]), sum(log2_shapes[j])
    for node in neighbors[j]  # rm multi-edge
        idxj_n = indexof(node, neighbors[j])
        idxn_j = indexof(j, neighbors[node])
        if node in neighbors[i]
            idxi_n = indexof(node, neighbors[i])
            idxn_i = indexof(i, neighbors[node])
            log2_shapes[i][idxi_n] += log2_shapes[j][idxj_n]
            log2_shapes[node][idxn_i] += log2_shapes[node][idxn_j]
            deleteat!(log2_shapes[node], idxn_j)
            deleteat!(neighbors[node], idxn_j)
        else
            push!(log2_shapes[i], log2_shapes[j][idxj_n])
            push!(neighbors[i], node)
            neighbors[node][idxn_j] = i
        end
    end
    add!(edge_info, [i, neighbors[i]...], log2_shapes, neighbors, strategy)
    log2_tc = log2_shapei + log2_shapeij + log2_shapej
    log2_sc = log2_shapei + log2_shapej
    # completely remove j
    empty!(log2_shapes[j])
    empty!(neighbors[j])
    return log2_tc, log2_sc
end

"""
    order_greedy(tn::TensorNetworkLayout; strategy="min_dim")

Compute greedy order, return the time and space complexities.
"""
function order_greedy(tn::TensorNetworkLayout; strategy="min_dim", edge_pool=collect(edges(tn)))
    order = SimpleEdge{Int}[]
    log2_shapes = deepcopy(tn.log2_shapes)
    neighbors = deepcopy(tn.graph.fadjlist)
    edge_info = edgeinfo(edges(tn), log2_shapes, neighbors, strategy)
    log2_tcs = Float64[] # time complexity
    log2_scs = Float64[] # space complexity

    while !isempty(edge_pool)
        edge = select(edge_info, strategy, neighbors, edge_pool)
        push!(order, edge)
        log2_tc_step, log2_sc_step = abstract_contract_edge!(edge, log2_shapes, edge_info, neighbors, strategy)
        push!(log2_tcs, log2_tc_step)
        push!(log2_scs, log2_sc_step)
        deleteat!(edge_pool, indexof(edge, edge_pool))

        i, j = src(edge), dst(edge)
        for (l, el) in enumerate(edge_pool)
            if j == src(el) || j == dst(el)
                k = src(el) == j ? dst(el) : src(el)
                edge_pool[l] = uedge(i, k)
            end
        end
        edge_pool = unique(edge_pool)
    end
    return log2_tcs, log2_scs, order
end

function abstract_contract(tn::TensorNetworkLayout, order)
    @assert check_healthy(tn) "tensor network shape mismatch!"
    log2_tcs = Float64[]
    log2_scs = Float64[]
    log2_shapes = deepcopy(tn.log2_shapes)
    neighbors = deepcopy(tn.graph.fadjlist)
    edge_info = edgeinfo(edges(tn), log2_shapes, neighbors, "min_dim")
    maxsize = 0.0
    for edge in order
        log2_tc, log2_sc = abstract_contract_edge!(edge, log2_shapes, edge_info, neighbors, "min_dim")
        push!(log2_tcs, log2_tc)
        push!(log2_scs, compute_log2_sc(log2_shapes))
        maxsize = max(maxsize, maximum(sum.(log2_shapes)))
    end
    return log2_tcs, log2_scs, maxsize
end

function check_healthy(tn::TensorNetworkLayout)
    res = true
    for i in vertices(tn.graph)
        for j in neighbors(tn.graph, i)
            res = res && (edgesize(tn, i, j) == edgesize(tn, j, i))
        end
    end
    return res
end

log2size(s) = isempty(s) ? -Inf : sum(s)
compute_log2_sc(log2_shapes) = (isempty(log2_shapes) || all(isempty, log2_shapes)) ? 0.0 : log2sumexp2(log2size.(log2_shapes))

edgesize(tn::TensorNetworkLayout, m::Int, n::Int) = tn.log2_shapes[n][indexof(m, neighbors(tn.graph, n))]
