export LabeledTensor

# abstractions
abstract type AbstractTensor{T, N} end

struct LabeledTensor{T,N,AT<:AbstractArray{T,N}, LT, MT} <: AbstractTensor{T, N}
    array::AT
    labels::Vector{LT}
    meta::MT
end
Base.ndims(t::LabeledTensor) = ndims(t.array)
LinearAlgebra.norm(t::LabeledTensor, p::Real=2) = norm(t.array, p)

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
Base.copy(t::LabeledTensor) = LabeledTensor(copy(t.array), t.labels)
Base.similar(::Type{<:LabeledTensor}, arr::AbstractArray, labels::AbstractVector, meta=nothing) = LabeledTensor(arr, labels, meta)
Base.similar(::LabeledTensor, arr::AbstractArray, labels::AbstractVector, meta=nothing) = LabeledTensor(arr, labels, meta)
LinearAlgebra.rmul!(t::LabeledTensor, factor) = (rmul!(t.array, factor); t)

function Base.show(io::IO, lt::LabeledTensor)
    print(io, "$(typeof(lt).name){$(eltype(lt.array))}($(join(lt.labels, ", ")))")
end

function Base.show(io::IO, ::MIME"plain/text", lt::LabeledTensor)
    Base.show(io, lt)
end

function mul_dim(t::LabeledTensor, m::AbstractMatrix; dim::Int)
    data = t.array
    iA = ntuple(i->i, ndims(data))
    iB = (dim, -dim)
    iC = ntuple(i->i==dim ? -dim : i, ndims(data))
    LabeledTensor(tensorcontract(iA, data, iB, m, iC), t.labels)
end

struct PlotMeta
    loc::Tuple{Float64, Float64}
    name::String
end
merge_meta(m1::PlotMeta, m2::PlotMeta) = PlotMeta((m1.loc .+ m2.loc) ./ 2, m1.name*m2.name)
merge_meta(m1::Nothing, m2::Nothing) = nothing
dispmeta(m::PlotMeta) = m.name


