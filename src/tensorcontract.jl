export tensorcontract, LabeledTensor, TensorNetwork, contract_label!, PlotMeta, contract, ContractionTree
export contract_tree

function _align_eltypes(xs::AbstractArray...)
    T = promote_type(eltype.(xs)...)
    return map(x->eltype(x)==T ? x : T.(x), xs)
end

function _align_eltypes(xs::AbstractArray{T}...) where T
    xs
end

# batched routines
@inline _indexpos(iAs, i)::Int = findfirst(==(i), iAs)
_isunique(x) = length(unique(x)) == length(x)

# can be used in either static or dynamic invoke
@noinline function analyse_dim(iAs, iBs, iOuts)
    # check indices
    @assert _isunique(iAs) "indices in A matrix is not unique $iAs"
    @assert _isunique(iBs) "indices in B matrix is not unique $iBs"
    @assert _isunique(iOuts) "indices in C matrix is not unique $iOuts"
    allinds = [iAs..., iBs..., iOuts...]
    @assert all(i -> count(==(i), allinds) == 2, allinds) "indices does not appear in pairs! got $iAs, $iBs and $iOuts"

    mdims = setdiff(iAs, iBs)
    ndims = setdiff(iBs, iAs)
    kdims = iAs ∩ iBs
    iABs = mdims ∪ ndims
    lA = mdims ∪ kdims
    lB = kdims ∪ ndims
    lOut = mdims ∪ ndims
    pA = indexin(lA, iAs)
    pB = indexin(lB, iBs)
    pOut = indexin(iOuts, lOut)
    pA, pB, pOut, length(kdims)
end

@noinline function analyse_size(pA, sA, pB, sB, nc)
    nA = length(sA)
    nB = length(sB)
    sA1 = Int[sA[pA[i]] for i=1:nA-nc]
    sB2 = Int[sB[pB[i]] for i=nc+1:nB]
    K = mapreduce(i->sB[pB[i]], *, 1:nc, init=1)
    return prod(sA1), K, prod(sB2), [sA1..., sB2...]
end

function tensorcontract(iAs, A::AbstractArray, iBs, B::AbstractArray, iOuts)
    A, B = _align_eltypes(A, B)
    pA, pB, pOut, nc = analyse_dim([iAs...], [iBs...], [iOuts...])
    M, K, N, sOut = analyse_size(pA, size(A), pB, size(B), nc)

    Apr = reshape(_conditioned_permutedims(A, pA), M, K)
    Bpr = reshape(_conditioned_permutedims(B, pB), K, N)
    AB = Apr * Bpr
    AB = _conditioned_permutedims(reshape(AB, sOut...), (pOut...,))
end

function _conditioned_permutedims(A::AbstractArray{T,N}, perm) where {T,N}
    if any(i-> (@inbounds perm[i]!=i), 1:N)
        return permutedims(A, (perm...,))
    else
        return A
    end
end

# abstractions
struct LabeledTensor{T,N,AT<:AbstractArray{T,N}, LT, MT}
    array::AT
    labels::Vector{LT}
    meta::MT
end
Base.ndims(t::LabeledTensor) = ndims(t.array)
LinearAlgebra.norm(t::LabeledTensor, p::Real=2) = norm(t.array)

function LabeledTensor(tensor::AbstractArray, labels::AbstractVector)
    @assert ndims(tensor) == length(labels) "dimension of tensor $(ndims(tensor)) != number of labels $(length(labels))"
    LabeledTensor(tensor, labels, nothing)
end

function Base.:(*)(A::LabeledTensor, B::LabeledTensor)
    labels_AB = setdiff(A.labels, B.labels) ∪ setdiff(B.labels, A.labels)
    LabeledTensor(tensorcontract(A.labels, A.array, B.labels, B.array, labels_AB), labels_AB, merge_meta(A.meta, B.meta))
end

function Base.isapprox(a::LabeledTensor, b::LabeledTensor; kwargs...)
    isapprox(a.array, b.array; kwargs...) && a.labels == b.labels
end
Base.size(t::LabeledTensor) = Base.size(t.array)

struct TensorNetwork{T,LT}
    tensors::Vector{LabeledTensor{T,N,AT,LT} where {N, AT}}
end

function TensorNetwork(tensors)
    T = promote_type([eltype(x.array) for x in tensors]...)
    LT = promote_type([eltype(x.labels) for x in tensors]...)
    TensorNetwork((LabeledTensor{T,N,AT,LT} where {N,AT})[tensors...])
end

Base.copy(tn::TensorNetwork) = TensorNetwork(copy(tn.tensors))
Base.length(tn::TensorNetwork) = length(tn.tensors)

struct ContractionTree
    left
    right
    function ContractionTree(left::Union{Integer, ContractionTree}, right::Union{Integer, ContractionTree})
        new(left, right)
    end
end
function Base.getindex(ct::ContractionTree, i::Int)
     Base.@boundscheck i==1 || i==2
    i==1 ? ct.left : ct.right
end

function contract_tree(tn::TensorNetwork{T}, ctree; normalizer) where T
    if ctree isa Integer
        t = tn.tensors[ctree]
        if normalizer !== nothing
            normt = normalizer(t)
            return normt, LabeledTensor(rmul!(copy(t.array), T(1/normt)), t.labels)
        else
            return one(T), t
        end
    else
        normt1, t1 = contract_tree(tn, ctree[1]; normalizer=normalizer)
        normt2, t2 = contract_tree(tn, ctree[2]; normalizer=normalizer)
        normt = normt1 * normt2
        t = t1 * t2
        if normalizer !== nothing
            normt_ = normalizer(t)
            normt = normt * normt_
            rmul!(t.array, T(1/normt_))
        end
        return normt, t
    end
end

function contract(tn::TensorNetwork, ctree; normalizer=nothing)
    normt, t = contract_tree(tn, ctree; normalizer=normalizer)
    t.array .*= normt
    return t
end

function contract_label!(tn::TensorNetwork{T, LT}, label::LT) where {T, LT}
    ts = findall(x->label ∈ x.labels, tn.tensors)
    @assert length(ts) == 2 "number of tensors with the same label $label is not 2, find $ts"
    t1, t2 = tn.tensors[ts[1]], tn.tensors[ts[2]]
    tout = t1 * t2
    deleteat!(tn.tensors, ts)
    push!(tn.tensors, tout)
    return tout, t1.labels ∩ t2.labels
end

using Base.Cartesian
@generated function c2l(size::NTuple{N, Int}, c::NTuple{N,Int}) where N
    quote
        res = c[1]
        stride = size[1]
        @nexprs $(N-1) i->begin
            res += (c[i+1]-1) * stride
            stride *= size[i+1]
        end
        return res
    end
end

@generated function l2c(size::NTuple{N, Int}, l::Int) where N
    quote
        l -= 1
        @nexprs $(N-1) i->begin
            @inbounds l, s_i = divrem(l, size[i])
        end
        $(Expr(:tuple, [:($(Symbol(:s_, i))+1) for i=1:N-1]..., :(l+1)))
    end
end

function Base.show(io::IO, tn::TensorNetwork)
    print(io, "$(typeof(tn).name):\n  $(join(["$(t.meta === nothing ? i : dispmeta(t.meta)) => $t" for (i, t) in enumerate(tn.tensors)], "\n  "))")
end

function Base.show(io::IO, ::MIME"plain/text", tn::TensorNetwork)
    Base.show(io, tn)
end

function Base.show(io::IO, lt::LabeledTensor)
    print(io, "$(typeof(lt).name){$(eltype(lt.array))}($(join(lt.labels, ", ")))")
end

function Base.show(io::IO, ::MIME"plain/text", lt::LabeledTensor)
    Base.show(io, lt)
end

struct PlotMeta
    loc::Tuple{Float64, Float64}
    name::String
end
merge_meta(m1::PlotMeta, m2::PlotMeta) = PlotMeta((m1.loc .+ m2.loc) ./ 2, m1.name*m2.name)
merge_meta(m1::Nothing, m2::Nothing) = nothing
dispmeta(m::PlotMeta) = m.name

function adjacency_matrix(tn::TensorNetwork)
    indx = Int[]
    indy = Int[]
    data = Int[]
    for i=1:length(tn)
        for j=1:length(tn)
            if i!=j && any(l->l ∈ tn.tensors[i].labels, tn.tensors[j].labels)
                push!(indx, i)
                push!(indy, j)
                push!(data, 1)
            end
        end
    end
    return sparse(indx, indy, data)
end
