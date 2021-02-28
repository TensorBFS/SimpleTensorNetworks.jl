using Test
using SimpleTensorNetworks.ContractionOrder
using LightGraphs: SimpleEdge

@testset "edge info" begin
    egs = [SimpleEdge(1, 2), SimpleEdge(2, 3), SimpleEdge(3, 1), SimpleEdge(3, 4)]
    log2_shapes = [log2.((2,2)), log2.((2,3)), log2.((3,2,6)), log2.((6,))]
    neighbors = [[2,3], [1,3], [1,2,4], [3]]
    strategy = "max_reduce_tri"
    ei = edgeinfo(egs[1:3], log2_shapes, neighbors, strategy)
    @test ei.data[SimpleEdge(3, 1)] ≈ -24.0
    @test ei.data[SimpleEdge(1, 2)] ≈ -4
    @test ei.data[SimpleEdge(2, 3)] ≈ -18
    @test select(ei, strategy, neighbors, nothing) == SimpleEdge(3, 1)

    # add!, remove!
    ei2 = add!(copy(ei), [4], log2_shapes, neighbors, strategy)
    ei3 = edgeinfo(egs, log2_shapes, neighbors, strategy)
    for (k, v) in ei2.data
        @test ei3.data[k] ≈ v
    end

    ei4 = remove!(copy(ei2), [4])
    @test ei4.data == ei.data
end
