export rm_multiedges!, rm_degree12, to_simplified_tnet
using LightGraphs: SimpleGraph, add_edge!

function rm_multiedges!(tn::TensorNetwork)
    n = length(tn.tensors)
    for i=1:n
        ti = tn.tensors[i]
        for j=i+1:n
            tj = tn.tensors[j]
            if count(x->x ∈ ti.labels, tj.labels) > 1
                # rm multi edges
                tn.tensors[i], tn.tensors[j] = rm_multiedge(ti, tj)
            end
        end
    end
    return tn
end

function rm_multiedge(ti, tj, li::Tuple, lj::Tuple)
    ti, tj, li, lj = rm_multiedge(ti, tj, collect(li), collect(lj))
    ti, tj, (li...,), (lj...,)
end

function rm_multiedge(t1, t2)
    ti, li, tj, lj = t1.array, t1.labels, t2.array, t2.labels
    common_edges = li ∩ lj
    dimsi = indexin(common_edges, li)
    remsi = setdiff(1:length(li), dimsi)
    dimsj = indexin(common_edges, lj)
    remsj = setdiff(1:length(lj), dimsj)
    orderi = Int[(remsi ∪ dimsi)...]
    orderj = Int[(remsj ∪ dimsj)...]
    # permute
    ti = permutedims(ti, orderi)
    tj = permutedims(tj, orderj)
    li = li[orderi]
    lj = lj[orderj]
    # reshape
    ti = reshape(ti, size(ti)[1:length(remsi)]..., :)
    tj = reshape(tj, size(tj)[1:length(remsj)]..., :)
    li = li[1:length(remsi)+1]
    lj = lj[1:length(remsj)+1]
    return similar(t1, ti, li), similar(t2, tj, lj)
end

Base.similar(::Type{<:LabeledTensor}, arr::AbstractArray, labels::AbstractVector, meta=nothing) = LabeledTensor(arr, labels, meta)

function tn2graph(tn::TensorNetwork)
    n = length(tn.tensors)
    g = SimpleGraph(n)
    for i=1:n
        ti = tn.tensors[i]
        for j=i+1:n
            if any(x->x ∈ ti.labels, tn.tensors[j].labels)
                add_edge!(g, i, j)
            end
        end
    end
    return g
end

function rm_degree12(tn::TensorNetwork{T}) where T
    tn = copy(tn)
    n = length(tn)
    mask = ones(Bool, n)
    has_dangling_edges = true
    factor = one(T)
    while has_dangling_edges && length(tn) > 1
        has_dangling_edges = false
        for i=1:n
            mask[i] || continue
            ti = tn.tensors[i]
            if ndims(ti) <= 2
                has_dangling_edges = true
                mask[i] = false
                if ndims(ti) != 0
                    j = findfirst(k -> mask[k] && (ti.labels[1] ∈ tn.tensors[k].labels), 1:n)
                    # absorb i -> j
                    tn.tensors[j] = ti * tn.tensors[j]
                else
                    factor *= Array(tn.tensors[i].array)[]
                end
            end
        end
    end
    return factor, TensorNetwork(tn.tensors[mask])
end
