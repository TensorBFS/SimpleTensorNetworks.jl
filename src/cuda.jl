using .CUDA
using .CUDA: CuArray, @linearidx, GPUArrays, CUBLAS
using LinearAlgebra

export togpu

function togpu(tn::TensorNetwork)
    TensorNetwork(togpu.(tn.tensors))
end

function togpu(t::LabeledTensor)
    LabeledTensor(CuArray(t.array), t.labels, t.meta)
end

function genperm(I::NTuple{N}, perm::NTuple{N}) where N
    ntuple(d-> (@inbounds return I[perm[d]]), Val(N))
end

function LinearAlgebra.permutedims!(dest::GPUArrays.AbstractGPUArray, src::GPUArrays.AbstractGPUArray, perm)
    LinearAlgebra.permutedims!(dest::GPUArrays.AbstractGPUArray, src::GPUArrays.AbstractGPUArray, Tuple(perm))
end
function LinearAlgebra.permutedims!(dest::GPUArrays.AbstractGPUArray, src::GPUArrays.AbstractGPUArray, perm::NTuple)
    perm isa Tuple || (perm = Tuple(perm))
    size_dest = size(dest)
    size_src = size(src)
    CUDA.gpu_call(vec(dest), vec(src), perm; name="permutedims!") do ctx, dest, src, perm
        i = @linearidx src
        I = l2c(size_src, i)
        @inbounds dest[c2l(size_dest, genperm(I, perm))] = src[i]
        return
    end
    return reshape(dest, size(dest))
end

function LinearAlgebra.rmul!(a::StridedCuArray{<:CUBLAS.CublasFloat}, k::Number)
    vec(a) .*= k
    return a
end
