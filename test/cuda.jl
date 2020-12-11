using Test
using SimpleTensorNetworks
using CUDA
using SimpleTensorNetworks: c2l, l2c
using Random

CUDA.allowscalar(false)

@testset "c2l" begin
    for i=1:100
        shape = (4,rand(1:5),rand(1:7),5,19)
        target = ([rand(1:s) for s in shape]...,)
        @test c2l(shape, target) == LinearIndices(shape)[target...]
    end
    for i=1:100
        shape = (4,rand(1:5),rand(1:12),15,19)
        ci = CartesianIndices(shape)
        i = rand(1:prod(shape))
        @test l2c(shape, i) == ci[i].I
    end
end

@testset "permutedims" begin
    a = randn(rand(1:3, 20)...)
    A = CuArray(a)
    p = randperm(20)
    @test Array(permutedims(A, p)) ≈ permutedims(a, p)
end


@testset "tensor contract - GPU" begin
    A = zeros(Float64, 10, 32, 21);
    B = zeros(Float64, 32, 11, 5, 2, 41, 10);
    tA = LabeledTensor(A, [1,2,3])
    tB = LabeledTensor(B, [2,4,5,6,7,1])
    tOut = tA * tB
    tnet = TensorNetwork([tA, tB]) |> togpu

    tOut2, contracted_labels = contract_label!(tnet, 1)
    @test Array(tnet.tensors[].array) ≈ tOut.array
    @test Array(tnet.tensors[].array) ≈ Array(tOut2.array)
    @test contracted_labels == [1, 2]
end
