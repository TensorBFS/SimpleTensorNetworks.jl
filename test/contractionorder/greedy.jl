using Test, SimpleTensorNetworks.ContractionOrder
using LightGraphs
using LightGraphs: SimpleEdge
using SimpleTensorNetworks.ContractionOrder: abstract_contract_edge!, log2sumexp2

function get_shapes(edges, sizes, neighbors)
    log2_shapes = Vector{Float64}[]
    for i=1:length(neighbors)
        si = Float64[]
        for nb in neighbors[i]
            ie = findfirst(e->e == SimpleEdge(i=>nb) || e == SimpleEdge(nb=>i), edges)
            push!(si, sizes[ie])
        end
        push!(log2_shapes, si)
    end
    return log2_shapes
end

tolog2(shapes) = [log2.(x) for x in shapes]

@testset "abstract contract" begin
    es = [SimpleEdge(1, 2), SimpleEdge(2, 3), SimpleEdge(1, 3), SimpleEdge(3, 4)]
    log2_sizes = log2.([2, 3, 4, 5])
    neighbors = [[2,3], [3,1], [1,2,4], [3]]
    log2_shapes = get_shapes(es, log2_sizes, neighbors)
    strategy = "min_reduce"
    edge_info = edgeinfo(es, log2_shapes, neighbors, strategy)

    edge = SimpleEdge(2, 3)
    tc, sc = abstract_contract_edge!(edge, log2_shapes, edge_info,
        neighbors, strategy)
    @test edge_info.data[SimpleEdge(2, 4)] ≈ 8.0
    @test edge_info.data[SimpleEdge(1, 2)] ≈ 5.0
    @test log2_shapes == [[8], [8, 5], Int[], [5]] |> tolog2
    @test neighbors == [[2], [1, 4], Int[], [2]]
    @test tc ≈ log2(120)

    es = [SimpleEdge(1, 2), SimpleEdge(2, 3), SimpleEdge(1, 3), SimpleEdge(3, 4)]
    log2_sizes = log2.([2, 3, 4, 5])
    neighbors = [[2,3], [3,1], [1,2,4], [3]]
    log2_shapes = get_shapes(es, log2_sizes, neighbors)
    tn = TensorNetworkLayout(SimpleGraph(4, neighbors), log2_shapes)
    order = [SimpleEdge(1, 2),SimpleEdge(1, 3), SimpleEdge(1, 4)]
    tcs, scs = abstract_contract(tn, order)
    @test log2sumexp2(tcs) ≈ log2(89)
    @test maximum(scs) ≈ log2(12*5+12+5)
end

@testset "greedy order" begin
    g = random_regular_graph(10, 3)
    log2_shapes = [[2,2,2] for i=1:10] |> tolog2
    tn = TensorNetworkLayout(g, log2_shapes)
    tcs, scs, order = order_greedy(tn; strategy="min_dim")
    @test length(order) <= length(edges(g))
end

@testset "disconnected graph" begin
    n = 5
    g = SimpleGraph(n)
    add_edge!(g, 2, 3)
    tn = TensorNetworkLayout(g, [ones(degree(g, i)) for i=1:n])
    tc, sc, orders = order_greedy(tn)
    @test orders isa Vector
    tcs, scs = abstract_contract(tn, orders)
    @test maximum(tcs) == 1
end
