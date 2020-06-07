module BSplines

using IterTools
# using OffsetArrays

export BSpline, calc, gen_matrix, predict

struct BSpline
    p
    n
    t
end

function BSpline(p::Int, t::AbstractArray{T}) where T <: Real
    # number of knots t[i]
    n = size(t, 1)

    # extend knots by p on both sides of t
    tt = [repeat(t[1:1], p)..., t..., repeat(t[end:end], p)...]

    BSpline(p, n, tt)
end

function calc(b::BSpline, i::Int, p::Int, x::T) where T <: Real
    if (i + p + 1 > b.n + 2 * b.p) || (i < 1) || !(b.t[i] <= x <= b.t[i+p+1])
        return zero(T)
    elseif p === b.p && i === b.n + p - 1 && x === b.t[end]
        return one(T)
    end

    if p === 0
        if (i + 1 <= b.n + b.p) && (b.t[i] <= x < b.t[i+1])
            one(T)
        else
            zero(T)
        end
    else
        b1 = calc(b, i, p - 1, x)
        a1 = b1 === zero(T) ? zero(T) : (x - b.t[i]) / (b.t[i + p] - b.t[i])

        b2 = calc(b, i + 1, p - 1, x)
        a2 = b2 === zero(T) ? zero(T) : (b.t[i + p + 1] - x) / (b.t[i + p + 1] - b.t[i + 1])

        a1 * b1 + a2 * b2
    end
end

calc(b::BSpline, i::Int, x::T) where T <: Real = calc(b, i, b.p, x)

function gen_matrix(b::BSpline, xs::AbstractArray{T}) where T <: Real
    [
        calc(b, i, x) for x in xs, i in 1:(b.n + b.p - 1)
    ]
end

function predict(b::BSpline, w::AbstractArray{T}, x::T) where T <: Real
    parts = partition(b.t, 2, 1) |> collect
    k = findfirst(i -> first(i) <= x < last(i), parts)
    if isnothing(k)
        return zero(T)
    end
    sum(w[i] * calc(b, i, b.p, x) for i in (k - b.p):k)
end

predict(b::BSpline, x::T) where T <: Real = predict(b, ones(T, b.n + b.p - 1), x)

end  # module
