using Test
using LightGraphs
using SimpleTensorNetworks
using Random

@testset "greedy" begin
    include("edgeinfo.jl")
end

@testset "greedy" begin
    include("greedy.jl")
end

@testset "other" begin
    for seed in 1:10
        Random.seed!(3)
        g = random_regular_graph(10, 3)
        tn = TensorNetwork([LabeledTensor(randn(2,2,2), [e for e in edges(g) if i ∈ (e.src, e.dst)]) for i=1:10])
        tcs, scs, order = SimpleTensorNetworks.order_greedy(SimpleTensorNetworks.layout(tn); strategy="min_reduce")
        trees = SimpleTensorNetworks.build_trees(10, order)
        order_new = SimpleTensorNetworks.build_order(trees)
        #@test order == order_new
        tc, sc = abstract_contract(SimpleTensorNetworks.layout(tn), order)
        tc2, sc2 = abstract_contract(tn, trees)
        @test sort(tc) ≈ sort(tc2)
    end

    @testset "log2sumexp2" begin
        x = randn(10)
        @test log2(sum(exp2.(x))) ≈ log2sumexp2(x)
    end
end
