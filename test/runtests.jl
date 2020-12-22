using SimpleTensorNetworks
using Test

@testset "contract" begin
    include("tensorcontract.jl")
end

@testset "simplify" begin
    include("simplify.jl")
end

@testset "viz" begin
    include("viz.jl")
end

@testset "cuda" begin
    if Base.find_package("CUDA") !== nothing
        include("cuda.jl")
    end
end
