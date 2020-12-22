using Test
using SimpleTensorNetworks

@testset "rm multi edges" begin
    t1 = randn(2,3,4,5)
    l1 = ['a', 'b', 'c', 'd']
    t2 = randn(6,4,7,2,3)
    l2 = ['e', 'c', 'f', 'a', 'b']
    t1_, t2_ = SimpleTensorNetworks.rm_multiedge(LabeledTensor(t1, l1), LabeledTensor(t2, l2))
    @test t1_.array == reshape(permutedims(t1, (4,1,2,3)), 5, :)
    @test t2_.array == reshape(permutedims(t2, (1,3,4,5,2)), 6,7,:)
    @test t1_.labels == ['d', 'a']
    @test t2_.labels == ['e', 'f', 'a']
end

@testset "rm single nodes" begin
    tn = TensorNetwork([
        LabeledTensor(randn(2), ["a"]),
        LabeledTensor(randn(2,2,2,2,2), ["g", "c", "a", "k", "l"]),
        LabeledTensor(randn(2), ["c"]),
        LabeledTensor(randn(2,2,2), ["g", "k", "l"]),
        ])
    r1 = contract(tn, [[[1,2], 3], 4])
    factor, tn = rm_degree12(tn)
    @test length(tn) == 2
    @test tn.tensors[1].labels == ["g", "k", "l"]
    @test tn.tensors[2].labels == ["g", "k", "l"]
    r2 = contract(tn, [1,2])
    @test r1 ≈ r2
    @test factor == 1.0

    tn = TensorNetwork([
        LabeledTensor(randn(2), ["a"]),
        LabeledTensor(randn(2,2,2,2), ["g", "c", "a", "k"]),
        LabeledTensor(randn(2), ["c"]),
        LabeledTensor(randn(2,2), ["g", "k"]),
        ])
    r1 = contract(tn, [[[1,2], 3], 4])
    factor, tn = rm_degree12(tn)
    @test length(tn) == 0
    @test r1.array[] ≈ factor
end
