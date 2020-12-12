using Test
using SimpleTensorNetworks
using Compose, Viznet

@testset "viz" begin
    tn = TensorNetwork([LabeledTensor(randn(2,2), ['a', 'b']), LabeledTensor(randn(2,2), ['b', 'c']), LabeledTensor(randn(2,2), ['a', 'c'])])
    @test viz_tnet(tn) isa Context
end
