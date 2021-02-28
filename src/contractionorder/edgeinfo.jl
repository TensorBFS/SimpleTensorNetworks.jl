export EdgeInfo, edgeinfo, update!, add!, remove!, select

struct EdgeInfo{DT<:Dict}
    data::DT
end

function EdgeInfo()
    EdgeInfo(Dict{SimpleEdge{Int},Float64}())
end

function edgeinfo(edges, log2_shapes, neighbors, strategy)
    ei = EdgeInfo()
    for edge in edges
        update!(ei, edge, log2_shapes, neighbors, strategy)
    end
    return ei
end

Base.copy(ei::EdgeInfo) = EdgeInfo(copy(ei.data))

# strategy ∈ {'min_dim','max_reduce','min_dim_tri','max_reduce_tri'}
function update!(ei::EdgeInfo, edge::SimpleEdge, log2_shapes::Vector, neighbors, strategy)
    m, n = src(edge), dst(edge)
    idxm_n = findfirst(==(n), neighbors[m])
    log2_shapemn = log2_shapes[m][idxm_n]
    log2_shapem = sum(log2_shapes[m])
    log2_shapen = sum(log2_shapes[n])
    if strategy == "min_reduce" || strategy == "min_reduce_tri"
        value = exp2(log2_shapem + log2_shapen - 2*log2_shapemn)
    elseif strategy == "max_reduce" || strategy == "max_reduce_tri"
        loga = log2_shapem + log2_shapen - 2*log2_shapemn
        logb = log2sumexp2([log2_shapem, log2_shapen])
        value = exp2(loga) - exp2(logb)  # different with previous one!
    else
        value = 1.0
    end
    ei.data[edge] = value
    return ei
end

function add!(ei::EdgeInfo, nodes::Vector{Int}, log2_shapes::Vector, neighbors, strategy)
    edges = SimpleEdge{Int}[]
    for node in nodes
        for neighbor in neighbors[node]
            edge = uedge(neighbor, node)
            if !(edge ∈ edges)
                push!(edges, edge)
                update!(ei, edge, log2_shapes, neighbors, strategy)
            end
        end
    end
    return ei
end

function remove!(ei::EdgeInfo, nodes)
    for edge in keys(ei.data)
        if src(edge) in nodes || dst(edge) in nodes
            pop!(ei.data, edge)
        end
    end
    return ei
end

"""
    select(ei::EdgeInfo, strategy, neighbors, edge_pool=nothing)

Select an edge from edge_pool, considering the priority of data.
"""
function select(ei::EdgeInfo, strategy, neighbors, edge_pool=nothing)
    pool_edge_info = if edge_pool !== nothing
        Dict(edge=>ei.data[edge] for edge in edge_pool)
    else
        ei.data
    end

    min_value = minimum(values(pool_edge_info))
    min_edges = [edge for edge in keys(pool_edge_info) if pool_edge_info[edge] == min_value]
    if strategy == "min_dim_tri" || strategy == "max_reduce_tri"
        triangle_count = zeros(Int, length(min_edges))
        for (i, edge) in enumerate(min_edges)
            triangle_count[i] = length(neighbors[src(edge)] ∩ neighbors[dst(edge)])
        end
        edge = min_edges[rand(findall(==(maximum(triangle_count)), triangle_count))]
    else
        edge = rand(min_edges)
    end
    return edge
end

Base.keys(ei::EdgeInfo) = keys(ei.data)
Base.haskey(ei::EdgeInfo, ind) = haskey(ei.data, ind)
nremain(ei::EdgeInfo) = length(ei.data)

"""undirected edge"""
uedge(i, k) = i < k ? SimpleEdge(i, k) : SimpleEdge(k, i)