module SimpleTensorNetworks

using Requires
using LinearAlgebra

include("tensors.jl")
include("tensorcontract.jl")
include("simplify.jl")

function __init__()
    @require CUDA = "052768ef-5323-5732-b1bb-66c8b64840ba" include("cuda.jl")
    @require Viznet = "52a3aca4-6234-47fd-b74a-806bdf78ede9" include("viz.jl")
end

end
