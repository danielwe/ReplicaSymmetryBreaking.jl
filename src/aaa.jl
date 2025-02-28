#####
##### Adaptive AAA on [0, 1]
##### adapted from RationalFunctionApproximation.jl#640a25b
"""
    Barycentric(nodes, values, weights)

Barycentric representation of a rational function that's real-valued on the real line and 
interpolated at real nodes

# Arguments (`T<:AbstractFloat`)
- `nodes::AbstractVector{T}`: interpolation nodes
- `values::AbstractVector{T}`: values at the interpolation nodes
- `weights::AbstractVector{T}`: barycentric weights for each node
```
"""
struct Barycentric{T}
    nodes::Vector{T}
    values::Vector{T}
    weights::Vector{T}
    function Barycentric(
        nodes::AbstractVector{T},
        values::AbstractVector{T},
        weights::AbstractVector{T},
    ) where {T<:AbstractFloat}
        @assert length(nodes) == length(values) == length(weights)
        return new{T}(nodes, values, weights)
    end
end

# Constructor from callable with predetermined nodes and weights
function Barycentric(
    f::F, nodes::AbstractVector{T}, weights::AbstractVector{T}
) where {F,T<:AbstractFloat}
    @assert !(F <: AbstractVector)
    values = Vector{T}(undef, length(nodes))  # force eltype regardless of f
    map!(f, values, nodes)
    return Barycentric(nodes, values, weights)::Barycentric{T}
end

# Constructor from callable with predetermined nodes and test nodes
# function Barycentric(
#     f::F, nodes::AbstractVector{T}, testnodes::AbstractVector{T}
# ) where {F,T<:AbstractFloat}
#     @assert !(F <: AbstractVector)
#     values = Vector{T}(undef, length(nodes))  # force eltype regardless of f
#     testvalues = Vector{T}(undef, length(testnodes))  # force eltype regardless of f
#     map!(f, values, nodes)
#     map!(f, testvalues, testnodes)
#     Lt = (transpose(testvalues) .- values) ./ (transpose(testnodes) .- nodes)
#     weights = last_left_singular_vector!(Lt)
#     return Barycentric(nodes, values, weights)::Barycentric{T}
# end

(fapprox::Barycentric)(z) = evaluate(fapprox, z)

@inline evaluate(fapprox::Barycentric, z) = _evaluate(fapprox, z)  # shim for autodiff intercept

@inline function _evaluate(fapprox::Barycentric{T}, z) where {T}
    (; nodes, values, weights) = fapprox
    ret0 = zero(T) + zero(z)
    ret = if isinf(z)
        # If the sum of weights is nonzero, r has a finite limit at infinity. Otherwise,
        # we don't know---it may approach a constant or diverge, and if it diverges it will
        # do so with a phase that's some unknown multiple of the phase of z. We're not going
        # to calculate the 1/z expansion to arbitrary order to figure all this out, so we
        # return NaN.
        wsum = sum(weights)
        if iszero(wsum)
            oftype(ret0, isreal(z) ? NaN : complex(NaN, NaN))
        else
            oftype(ret0, dot(weights, values) / wsum)
        end
    else
        num, den, node_i = evaluate_core(fapprox, z)
        if node_i != 0
            # This branch is takine within a miniscule, mostly hypothetical neighborhood of
            # z_i (within subnormal range, which means exactly at z_i unless z_i is small
            # enough that eps(z_i) is in the subnormals---roughly abs(z_i) < 1e-292). Hence,
            # considering only the value of the interpolant, the sensible thing to do would
            # be to simply return f_i. However, within this neighborhood, evaluate_inner
            # gives us for free the num and den used to calculate the derivative at z_i,
            # so we might as well compute the linearization. The linear term almost
            # certainly won't make a difference in the returned value, but it lets autodiff
            # know what to return when the forward pass took this branch. (If we simply
            # returned f_i, autodiff would report a derivative of zero.) This saves us from
            # implementing a custom rule to get sound derivatives.
            z_i, f_i, w_i = nodes[node_i], values[node_i], weights[node_i]
            oftype(ret0, f_i + (z - z_i) * derivative_at_node(f_i, w_i, num, den))
        else
            (num / den)::typeof(ret0)
        end
    end
    return ret
end

@inline function evaluate_core((; nodes, values, weights)::Barycentric{T}, z) where {T}
    ret0 = zero(T) + zero(z)
    num, den = ret0, ret0
    node_i = 0
    # for i in eachindex(nodes, values, weights)
    @simd for i in eachindex(nodes, values, weights)
        dz = z - nodes[i]
        # z == z_i needs special casing, but don't break the loop; we'd rather SIMD the
        # common case than optimize the exceptional case

        # # we flush subnormals to zero to avoid overflow to Inf in weights[i] / dz; this is
        # # sufficient since aaa produces a normalized weights vector, so abs(weights[i]) <= 1
        # atnode = (abs(dz) < floatmin(typeof(dz)))
        # # take note of the index; using += lets us write this without node_i in the RHS,
        # # avoiding a loop-carried dependency and letting SIMD rip
        # node_i += atnode ? i : zero(i)
        # # we whould avoid division by zero even if the result would be discarded by ifelse,
        # # otherwise autodiff might pull back a NaN, so we use the short-circuiting ternary;
        # # fingers crossed that the compiler will optimize away the branch and SIMD
        #
        # # bonus: p = 0 turns num and den into quantities needed in the derivative
        # p = atnode ? ret0 : (weights[i] / dz)

        p = if abs(dz) < floatmin(typeof(dz))
            node_i = i
            zero(T) / oneunit(dz)
        else
            weights[i] / dz
        end
        num += p * values[i]
        den += p

        # if abs(dz) < floatmin(typeof(dz))
        #     node_i = i
        # else
        #     p = weights[i] / dz
        #     num += p * values[i]
        #     den += p
        # end
    end
    return num, den, node_i
end

@inline derivative_at_node(value, weight, num, den) = (num - value * den) / weight

# completely rewritten starting from BaryRational.jl#bba8867
# @inline function derivative(fapprox::Barycentric{T}, z) where {T}
#     (; nodes, values, weights) = fapprox
#     ret0 = zero(T) + zero(z)
#     ret = if isinf(z)
#         # If the sum of weights is nonzero, r has a finite limit at infinity, so the
#         # derivative vanishes. Otherwise, we have no idea, as discussed in the corresponding
#         # comment in `evaluate`, so we return NaN.
#         wsum = sum(weights)
#         iszero(wsum) ? oftype(ret0, isreal(ret0) ? NaN : complex(NaN, NaN)) : ret0
#     else
#         num, den, node_index = evaluate_core(fapprox, z)
#         if node_index != 0
#             f_i, w_i = @inbounds (values[node_index], weights[node_index])
#             oftype(ret0, derivative_at_node(f_i, w_i, num, den))
#         else
#             primal = num / den
#             dnum = ret0
#             @inbounds @simd for i in eachindex(nodes, values, weights)
#                 dz = z - nodes[i]
#                 dnum += weights[i] * (primal - values[i]) / (dz * dz)
#             end
#             (dnum / den)::typeof(ret0)
#         end
#     end
#     return ret
# end
#
# function EnzymeRules.augmented_primal(
#     config::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(evaluate)},
#     ::Type{<:Active},
#     fapprox::Union{Const{B},Duplicated{B},DuplicatedNoNeed{B}},
#     z::Union{Const,Active},
# ) where {B<:Barycentric}
#     needs_primal = EnzymeRules.needs_primal(config)
#     needs_shadow = EnzymeRules.needs_shadow(config)
#     primal, shadow = if needs_primal || needs_shadow
#         p_true = evaluate(fapprox.val, z.val)
#         p = needs_primal ? p_true : nothing
#         s = needs_shadow ? make_zero(p_true) : nothing
#         (p, s)
#     else
#         (nothing, nothing)
#     end
#     return EnzymeRules.AugmentedReturn(primal, shadow, nothing)
# end
#
# function EnzymeRules.reverse(
#     ::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(evaluate)},
#     shadow::Active,
#     _,  # tape not used
#     fapprox::Union{Const{B},Duplicated{B},DuplicatedNoNeed{B}},
#     z::Union{Const,Active},
# ) where {B<:Barycentric}
#     # NOTE: naive autodiff of `evaluate` will give an incorrect derivative wrt. z when z is
#     # equal to a node, since `evaluate` effectively uses a lookup table at these points,
#     # hence autodiff would return zero (the nodes are removable singularities). Sampling
#     # exactly at nodes is likely because both `aaa` and `quadgk` are biased towards fairly
#     # simple rational fractions of the interval. However, autodiff gives the correct
#     # derivatives wrt. fapprox. Hence, this rule uses a hybrid approach, calling a
#     # handwritten function for the derivative wrt. z and calling back into autodiff for the
#     # derivatives wrt. fapprox.
#     dy, fapproxv, zv = shadow.val, fapprox.val, z.val
#     if !(fapprox isa Const)
#         fwd, rev = autodiff_thunk(
#             ReverseSplitWidth(ReverseSplitNoPrimal, Val(1)),
#             Const{typeof(_evaluate)},
#             Active,
#             typeof(fapprox),
#             Const{typeof(zv)},
#         )
#         thunktape, _, _ = fwd(Const(_evaluate), fapprox, Const(zv))
#         rev(Const(_evaluate), fapprox, Const(zv), dy, thunktape)
#     end
#     dz = (z isa Const) ? nothing : (dy * derivative(fapproxv, zv))
#     return (nothing, dz)
# end

function aaa(f::F, lower::T, upper::T; kwargs...) where {F,T}
    return first(_aaa(f, lower, upper; kwargs...))
end

function _aaa(
    f::F, lower::T, upper::T;
    max_degree=2^6 - 1,
    tol::T=eps(T)^(25 // 26),          # 0.5 hex digits (Float64)
    acceptable_tol::T=tol^(21 // 25),  # 2.5 hex digits (Float64)
    required_tol::T=tol^(17 // 25),    # 4.5 hex digits (Float64)
    refinement=3,
    lookahead=2^3,
    verbose=false,
) where {F,T<:AbstractFloat}
    if !(lower < upper)
        # throw(ArgumentError(lazy"require lower < upper, got lower = $lower, upper = $upper"))
        throw(ArgumentError("require lower < upper, got lower = $lower, upper = $upper"))
    end
    fT(x::TT) where {TT} = convert(TT, f(x))::TT  # force type regardless of f inferability
    besterr = T(Inf)
    bestM = 1
    local bestnodes
    local besttestnodes
    local bestvalues
    local bestweights
    # initial nodes and values
    nodes = [lower, upper]
    values = map(fT, nodes)
    M = length(nodes)
    # initial refinement
    Nrefine = 16 - M
    testnodes = refine(nodes, Nrefine)
    testvalues = map(fT, testnodes)
    fmax = max(norm(values, Inf), norm(testvalues, Inf))  # norm(f, Inf) estimate
    while true
        # NOTE: we're redoing a lot of subtraction and division every time when we create Lt
        # from scratch, but since we can use the destructive svd! rather than the copying
        # svd, we save on net allocations compared to only computing the new columns/rows
        # and splicing them with the existing matrix. (I tried.)

        Lt = (transpose(testvalues) .- values) ./ (transpose(testnodes) .- nodes)
        weights = last_left_singular_vector_nozero!(Lt)  # don't need the junk-clearing version here

        fapprox = Barycentric(nodes, values, weights)
        err, j_max = findmax(eachindex(testvalues, testnodes)) do j
            abs(testvalues[j] - fapprox(testnodes[j]))
        end
        goodpoles = !any(poles(nodes, weights)) do node
            iszero(imag(node)) && (lower <= real(node) <= upper)
        end

        # if valid and best yet, save
        if goodpoles && (err < besterr)
            besterr = err
            bestM = M
            bestnodes = copy(nodes)
            besttestnodes = copy(testnodes)
            bestvalues = copy(values)
            bestweights = weights
        end

        # stopping criteria
        if besterr <= (tol * fmax)
            # verbose && @info lazy"""
            verbose && @info """
                aaa: success, desired tolerance met
                best relative error  = $(besterr / fmax) at degree $(bestM - 1)
                desired tolerance    = $tol
                acceptable tolerance = $acceptable_tol
                required tolerance   = $required_tol"""
            break
        elseif ((M - bestM) >= lookahead) && (besterr <= (acceptable_tol * fmax))
            # verbose && @warn lazy"""
            verbose && @warn """
                aaa: stagnation, acceptable tolerance met
                best relative error  = $(besterr / fmax) at degree $(bestM - 1)
                desired tolerance    = $tol
                acceptable tolerance = $acceptable_tol
                required tolerance    = $required_tol"""
            break
        elseif M == (max_degree + 1)
            if besterr <= (required_tol * fmax)
                # verbose && @warn lazy"""
                verbose && @warn """
                    aaa: reached max degree = $max_degree, required tolerance met
                    best relative error  = $(besterr / fmax) at degree $(bestM - 1)
                    desired tolerance    = $tol
                    acceptable tolerance = $acceptable_tol
                    required tolerance   = $required_tol"""
            else
                # error(lazy"""
                # @error lazy"""
                @error """
                    aaa: reached max degree = $max_degree, tolerances not met
                    best relative error  = $(besterr / fmax) at degree $(bestM - 1)
                    desired tolerance    = $tol
                    acceptable tolerance = $acceptable_tol
                    required tolerance   = $required_tol"""
                # )
            end
            break
        end

        # continuing; add the worst test point to the nodes
        # NOTE: If Nrefine were constant we could calculate i_new_sorted, but it's not.
        newnode, newvalue = testnodes[j_max], testvalues[j_max]
        i_new_sorted = searchsortedfirst(nodes, newnode)  # maintain sortedness for refinement
        insert!(nodes, i_new_sorted, newnode)
        insert!(values, i_new_sorted, newvalue)
        M += 1

        # refine the subdivided segment
        # NOTE: This is different from the original algorithm: we only refine the segment
        # subdivided by newnode, keeping the existing refinement elsewhere and thus reusing
        # most f evals we already computed
        if Nrefine > refinement
            Nrefine -= 1
        end
        newsegments = @view nodes[(i_new_sorted-1):(i_new_sorted+1)]
        newtestnodes = refine(newsegments, Nrefine)
        newtestvalues = map(fT, newtestnodes)
        fmax = max(fmax, norm(newtestvalues, Inf))

        # A: replace existing test nodes within the refined segments
        j_start = searchsortedfirst(testnodes, first(newsegments))
        j_stop = (j_start - 1) + @views searchsortedlast(testnodes[j_start:end], last(newsegments))
        splice!(testnodes, j_start:j_stop, newtestnodes)
        splice!(testvalues, j_start:j_stop, newtestvalues)

        # B: keep all existing test nodes (seems to perform worse)
        # deleteat!(testnodes, j_max)
        # deleteat!(testvalues, j_max)
        # j = 0
        # @views for (newtestnode, newtestvalue) in zip(newtestnodes, newtestvalues)
        #     jrange = j .+ searchsorted(testnodes[(j+1):end], newtestnode)
        #     splice!(testnodes, jrange, newtestnode)
        #     splice!(testvalues, jrange, newtestvalue)
        # end
    end
    # TODO: throw proper error if best* undefined
    return Barycentric(bestnodes, bestvalues, bestweights), besttestnodes
end

@inline function refine(nodes::AbstractVector{T}, N::Int) where {T}  # S is assumed sorted
    Base.require_one_based_indexing(nodes)
    M = length(nodes) - 1
    @assert M > 0
    testnodes = Vector{T}(undef, N * M)
    for i in 1:M
        stop = i * N
        testnodes[(stop-(N-1)):stop] .= range(nodes[i], nodes[i+1], N + 2)[(begin+1):(end-1)]
    end
    return testnodes
end

@inline last_left_singular_vector!(A) = widesvd!(A).U[:, end]
@inline last_left_singular_vector_nozero!(A) = widesvd_nozero!(A).U[:, end]

@inline function widesvd!(A)
    SVD = widesvd_nozero!(A)
    fill!(A, zero(eltype(A)))  # clear the junk so custom rule can be truly correct
    return SVD
end

@inline function widesvd_nozero!(A)
    SVD = svd!(A; full=false)  #, alg=QRIteration())
    LinearAlgebra.checksquare(SVD.U)
    return SVD
end

@inline function poles(nodes::Vector{T}, weights::AbstractVector{T}) where {T}
    nonzero = findall(!iszero, weights)
    m = length(nonzero) + 1
    B = zeros(T, m, m)
    for i in 2:m
        B[i, i] = oneunit(T)
    end
    E = zeros(T, m, m)
    E[2:m, 1] .= oneunit(T)
    for i in 2:m
        E[1, i] = weights[nonzero[i-1]]
        E[i, i] = nodes[nonzero[i-1]]
    end
    return filter(isfinite, eigvals!(E, B))
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(aaa)},
    ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
    f::Annotation,
    a::Const{T},
    b::Const{T};
    kwargs...,
) where {T}
    needs_primal = EnzymeRules.needs_primal(config)
    needs_shadow = EnzymeRules.needs_shadow(config)
    reverse_needs_nodes = !(f isa Const)
    primal, shadow, tape = if needs_primal || needs_shadow || reverse_needs_nodes
        # under either of these conditions the primal is needed for _something_
        p_true, tn = _aaa(f.val, a.val, b.val; kwargs...)
        p = needs_primal ? p_true : nothing
        s = needs_shadow ? make_zero(p_true) : nothing
        # t = reverse_needs_nodes ? (p_true.nodes, tn) : (nothing, nothing)
        t = reverse_needs_nodes ? (p_true.nodes, p_true.weights) : (nothing, nothing)
        # NOTE: the tape always includes the shadow if it exists, regardless of whether it's
        # needed in the reverse pass---we keep it handy so we can zero it in the end
        # XXX: put f on tape as workaround for Enzyme.jl#2304
        (p, s, (s, t..., f))
    else
        (nothing, nothing, (nothing, nothing, nothing, f))
    end
    return EnzymeRules.AugmentedReturn(primal, shadow, tape)
end

function EnzymeRules.reverse(
    ::EnzymeRules.RevConfigWidth{1},
    ::Const{typeof(aaa)},
    ::Type{RT},
    tape::Tuple{Barycentric{T},Vector{T},Vector{T},F},
    farg::F,
    a::Const{T},
    b::Const{T};
    kwargs...,
) where {RT<:Union{Duplicated,DuplicatedNoNeed},F<:Union{Active,Duplicated,MixedDuplicated},T}
    shadow, nodes, testnodes = tape
    # XXX: workaround for Enzyme.jl#2304: would like to use the primal from the tape and
    # accumulate into the shadow from the argument, but they often contain values with
    # mismatched shapes
    ftape = last(tape)
    # f = mergeannotated(ftape, farg)::F
    f = ftape
    dupnodes = Duplicated(nodes, make_zero(nodes))
    duptestnodes = Duplicated(testnodes, make_zero(testnodes))
    fwd, rev = autodiff_thunk(
        ReverseSplitNoPrimal,
        Const{Type{Barycentric}},
        Duplicated,
        F,
        typeof(dupnodes),
        typeof(duptestnodes),
    )
    thunktape, _, thunkshadow = fwd(Const(Barycentric), f, dupnodes, duptestnodes)
    thunkshadow.nodes .= shadow.nodes
    thunkshadow.values .= shadow.values
    thunkshadow.weights .= shadow.weights
    df = first(only(rev(Const(Barycentric), f, dupnodes, duptestnodes, thunktape)))
    make_zero!(shadow)
    return (df, nothing, nothing)
end

# mergeannotated(p::A, ::A) where {A<:Union{Const,Active}} = p
# mergeannotated(p::A, d::A) where {A<:Duplicated} = Duplicated(p.val, d.dval)
# mergeannotated(p::A, d::A) where {A<:MixedDuplicated} = MixedDuplicated(p.val, d.dval)

# function EnzymeRules.augmented_primal(
#     config::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(last_left_singular_vector!)},
#     ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
#     A::Duplicated;
# )
#     SVD = widesvd!(A.val)
#     needs_primal = EnzymeRules.needs_primal(config)
#     needs_shadow = EnzymeRules.needs_shadow(config)
#     primal, shadow = if needs_primal || needs_shadow
#         u = SVD.U[:, end]
#         p = needs_primal ? u : nothing
#         s = if needs_primal && needs_shadow
#             make_zero(u)
#         elseif needs_shadow
#             # bit of microoptimization: if primal isn't needed we can use it as shadow
#             make_zero!(u)
#             u
#         else
#             nothing
#         end
#         (p, s)
#     else
#         (nothing, nothing)
#     end
#     tape = (SVD, shadow)
#     return EnzymeRules.AugmentedReturn(primal, shadow, tape)
# end
#
# function EnzymeRules.reverse(
#     ::EnzymeRules.RevConfigWidth{1},
#     ::Const{typeof(last_left_singular_vector!)},
#     ::Type{<:Union{Duplicated,DuplicatedNoNeed}},
#     tape::Tuple{Any,Any},
#     A::Duplicated;
# )
#     @views begin
#         SVD, shadow = tape
#         @assert !isnothing(shadow)
#
#         du = shadow
#         dA = A.dval
#
#         # NOTE: outdated names are assigned nothing to prevent reading overwritten memory
#
#         # partition singular parts into those corresponding to du vs. the rest
#         U_trunc, u = SVD.U[:, begin:(end-1)], SVD.U[:, end]
#         S_trunc, σ = SVD.S[begin:(end-1)], SVD.S[end]
#         Vt_trunc, vt = SVD.Vt[begin:(end-1), :], SVD.Vt[end:end, :]
#         SVD = nothing
#
#         # we don't need normalized right-singular vectors, only scaled ones
#         SVt_trunc = (Vt_trunc .*= S_trunc)  # A .* v == Diagonal(v) * A
#         Vt_trunc = nothing
#
#         σvt = (vt .*= σ)
#         vt = nothing
#
#         # we no longer need the singular values, only the pullback eigenvalues wrt. du
#         D = let σsq = σ^2
#             S_trunc .= inv.(S_trunc .^ 2 .- σsq)
#         end
#         S_trunc = nothing
#
#         # common factor: eigenvalue-scaled shadow-eigenvector overlap vector
#         DUtdu = let Utdu = transpose(U_trunc) * du  # the only fresh allocation---seems unavoidable
#             Utdu .*= D  # A .* v == Diagonal(v) * A
#         end
#
#         # Term 1: overwrite, don't accumulate, since primal zeroes A
#         UDUtdu = mul!(du, U_trunc, DUtdu)
#         du = nothing
#
#         dA .= σvt .* UDUtdu  # b' .* a == a * b': outer product
#
#         # Term 2: accumulate onto term 1
#         dutUDSVt = mul!(σvt, transpose(DUtdu), SVt_trunc)
#         σvt = nothing
#
#         dA .+= dutUDSVt .* u  # b' .* a == a * b': outer product
#
#         # Clear shadow before returning
#         make_zero!(shadow)
#     end
#     return (nothing,)
# end
