using .Viznet
using .Viznet.Compose
using SparseArrays

export viz_tnet

function viz_tnet(tnet::TensorNetwork; r=0.25/sqrt(length(tnet.tensors)+1), show_edgeindex=false,
        node_fontsize=100pt/sqrt(length(tnet.tensors)+1),
        edge_fontsize=200pt/sqrt(length(tnet.tensors)+1),
        labels=1:length(tnet),
        locs=spring_layout(tnet),
    )
    nt = length(tnet.tensors)
    nb = nodestyle(:default, fill("white"), stroke("black"), linewidth(2mm/sqrt(length(tnet.tensors)+1)); r=r)
    eb = bondstyle(:default, linewidth(4mm/sqrt(length(tnet.tensors)+1)), stroke("skyblue"))
    tb1 = textstyle(:default, fontsize(node_fontsize))
    tb2 = textstyle(:default, fontsize(edge_fontsize))
    compose(Compose.context(r, r, 1-2r, 1-2r), canvas() do
        for (t, loc, label) in zip(tnet.tensors, locs, labels)
            nb >> loc
            if !isempty(label)
                tb1 >> (loc, string(label))
            end
        end
        for i=1:nt
            for j=i+1:nt
                li = tnet.tensors[i].labels
                lj = tnet.tensors[j].labels
                loci, locj = locs[i], locs[j]
                common_labels = li ∩ lj
                if !isempty(common_labels)
                    eb >> (loci, locj)
                    show_edgeindex && tb2 >> ((loci .+ locj) ./ 2, join(common_labels, ", "))
                end
            end
        end
    end)
end

function Base.show(io::IO, mime::MIME"text/html", tnet::TensorNetwork)
    show(io, mime, viz_tnet(tnet))
end

# copied from LightGraphs
function spring_layout(tn::TensorNetwork,
                       locs_x=2*rand(length(tn)).-1.0,
                       locs_y=2*rand(length(tn)).-1.0;
                       C=2.0,
                       MAXITER=100,
                       INITTEMP=2.0)

    nvg = length(tn)
    adj_matrix = adjacency_matrix(tn)

    # The optimal distance bewteen vertices
    k = C * sqrt(4.0 / nvg)
    k² = k * k

    # Store forces and apply at end of iteration all at once
    force_x = zeros(nvg)
    force_y = zeros(nvg)

    # Iterate MAXITER times
    @inbounds for iter = 1:MAXITER
        # Calculate forces
        for i = 1:nvg
            force_vec_x = 0.0
            force_vec_y = 0.0
            for j = 1:nvg
                i == j && continue
                d_x = locs_x[j] - locs_x[i]
                d_y = locs_y[j] - locs_y[i]
                dist²  = (d_x * d_x) + (d_y * d_y)
                dist = sqrt(dist²)

                if !( iszero(adj_matrix[i,j]) && iszero(adj_matrix[j,i]) )
                    # Attractive + repulsive force
                    # F_d = dist² / k - k² / dist # original FR algorithm
                    F_d = dist / k - k² / dist²
                else
                    # Just repulsive
                    # F_d = -k² / dist  # original FR algorithm
                    F_d = -k² / dist²
                end
                force_vec_x += F_d*d_x
                force_vec_y += F_d*d_y
            end
            force_x[i] = force_vec_x
            force_y[i] = force_vec_y
        end
        # Cool down
        temp = INITTEMP / iter
        # Now apply them, but limit to temperature
        for i = 1:nvg
            fx = force_x[i]
            fy = force_y[i]
            force_mag  = sqrt((fx * fx) + (fy * fy))
            scale      = min(force_mag, temp) / force_mag
            locs_x[i] += force_x[i] * scale
            locs_y[i] += force_y[i] * scale
        end
    end

    # Scale to unit square
    min_x, max_x = minimum(locs_x), maximum(locs_x)
    min_y, max_y = minimum(locs_y), maximum(locs_y)
    function scaler(z, a, b)
        2.0*((z - a)/(b - a)) - 1.0
    end
    map!(z -> scaler(z, min_x, max_x), locs_x, locs_x)
    map!(z -> scaler(z, min_y, max_y), locs_y, locs_y)

    return [((x+1)/2, (y+1)/2) for (x, y) in zip(locs_x, locs_y)]
end
