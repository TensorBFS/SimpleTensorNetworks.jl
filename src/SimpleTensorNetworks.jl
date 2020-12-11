module SimpleTensorNetworks

using Requires
using LinearAlgebra

include("tensorcontract.jl")

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
end

end
