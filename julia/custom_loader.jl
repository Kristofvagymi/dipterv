using Random: AbstractRNG, shuffle!, GLOBAL_RNG

struct CustomDataLoader{D,R<:AbstractRNG}
    data::D
    batchsize::Int
    nobs::Int
    partial::Bool
    imax::Int
    indices::Vector{Int}
    shuffle::Bool
    rng::R
    mshoot_len::Int
end

function CustomDataLoader(data; batchsize=1, shuffle=false, partial=true, rng=GLOBAL_RNG, mshoot_len = 2)
    batchsize > 0 || throw(ArgumentError("Need positive batchsize"))

    n = _nobs(data)
    if n < batchsize
        @warn "Number of observations less than batchsize, decreasing the batchsize to $n"
        batchsize = n
    end
    imax = partial ? n : n - batchsize + 1
    CustomDataLoader(data, batchsize, n, partial, imax, [1:n;], shuffle, rng, mshoot_len)
end

Flux.Data.@propagate_inbounds function Base.iterate(d::CustomDataLoader, i=0)     # returns data in d.indices[i+1:i+batchsize]
    i >= d.imax && return nothing
    if d.shuffle && i == 0
        shuffle!(d.rng, d.indices)
    end
    nexti = min(i + d.batchsize, d.nobs)
    ids = d.indices[i+1:nexti]

    modifiedIds = []
    for id in ids
        id = min(id, d.imax - mshoot_len + 1 )

        for j in id:1:id + d.mshoot_len - 1
            if j >= d.imax
                push!(modifiedIds, id)
            else
                push!(modifiedIds, j)
            end
        end
    end
    batch = _getobs(d.data, modifiedIds)
    return (batch, nexti)
end

function Base.length(d::CustomDataLoader)
    n = d.nobs / d.batchsize
    d.partial ? ceil(Int,n) : floor(Int,n)
end

_nobs(data::AbstractArray) = size(data)[end]

function _nobs(data::Union{Tuple, NamedTuple})
    length(data) > 0 || throw(ArgumentError("Need at least one data input"))
    n = _nobs(data[1])
    for i in keys(data)
        ni = _nobs(data[i])
        n == ni || throw(DimensionMismatch("All data inputs should have the same number of observations, i.e. size in the last dimension. " *
            "But data[$(repr(first(keys(data))))] ($(summary(data[1]))) has $n, while data[$(repr(i))] ($(summary(data[i]))) has $ni."))
    end
    return n
end

_getobs(data::AbstractArray, i) = data[ntuple(i -> Colon(), Val(ndims(data) - 1))..., i]
_getobs(data::Union{Tuple, NamedTuple}, i) = map(Base.Fix2(_getobs, i), data)

Base.eltype(::CustomDataLoader{D}) where D = D
