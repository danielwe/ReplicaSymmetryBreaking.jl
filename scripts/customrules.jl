using Enzyme

@noinline f(x) = sum(abs2, x)
@noinline f_custom(x) = sum(abs2, x)

g(x) = cos(f(x))
g_custom(x) = cos(f_custom(x))

function dg_deferred!(dx, x)
    make_zero!(dx)
    autodiff_deferred(Reverse, g, Active, Duplicated(x, dx))
    return nothing
end

function dg_custom_deferred!(dx, x)
    make_zero!(dx)
    autodiff_deferred(Reverse, g_custom, Active, Duplicated(x, dx))
    return nothing
end

function EnzymeRules.augmented_primal(
    config::EnzymeRules.Config, f::Const{typeof(f_custom)}, ::Type{<:Active}, x::Duplicated
)
    tape = EnzymeRules.overwritten(config)[2] ? copy(x.val) : nothing
    primal = EnzymeRules.needs_primal(config) ? f.val(x.val) : nothing
    return EnzymeRules.AugmentedReturn(primal, nothing#=shadow=#, tape)
end

function EnzymeRules.reverse(
    config::EnzymeRules.Config,
    ::Const{typeof(f_custom)},
    dret::Active,
    tape,
    x::Duplicated,
)
    xval = EnzymeRules.overwritten(config)[2] ? tape : x.val
    x.dval .= (2dret.val) .* xval
    x.dval .^= 2  # Deliberate bug as signature of custom rule 🐛
    return (nothing,)
end

x = [2.0]
dx = make_zero(x)
dx_custom = make_zero(x)

v = first(onehot(x))
hv = make_zero(v)
hv_custom = make_zero(v)

# primals
@show g(x)
@show g_custom(x)

# gradients
dg_deferred!(dx, x)
@show dx

dg_custom_deferred!(dx_custom, x)
@show dx_custom

# hvps
autodiff(Forward, dg_deferred!, Const, Duplicated(dx, hv), Duplicated(x, v))
@show hv

autodiff(
    Forward,
    dg_custom_deferred!,
    Const,
    Duplicated(dx_custom, hv_custom),
    Duplicated(x, v),
)
@show hv_custom

###

# f(x) = sqrt(sum(abs2, x))
#
# function df_deferred!(dx, x)
#     make_zero!(dx)
#     autodiff_deferred(Reverse, f, Active, Duplicated(x, dx))
#     return nothing
# end
#
# x = [2.0]
# dx = make_zero(x)
# v = first(onehot(x))
# hv = make_zero(v)
# autodiff(
#     Forward,
#     df_deferred!,
#     Const,
#     Duplicated(dx, hv),
#     Duplicated(x, v),
# )
# @show hv
