### objective function
function objective(
    u::AbstractVector{T},
    problem::AbstractDisorderProblem{T},
    quad::AbstractQuadrature{T},
    # op::Orderparameters{T},  # TODO: should we preallocate op given we don't preallocate anything else now? probably not?
) where {T}
    # updateorderparameters!(op, problem, u; check=true, checksums=false)
    op = orderparameters(problem, u)
    return -freeenergy(problem, quad, op)
end

function nanobjective(
    u,
    problem,
    quad,
    # op,
    warn...,
)
    try
        # return objective(u, problem, quad, op)
        return objective(u, problem, quad)
    catch e
        checkRSBexceptions(e, warn...; actionmsg="Returning NaN") || rethrow()
        # return convert(Base.promote_op(objective, typeof.((u, problem, quad, op))...), NaN)
        return convert(Base.promote_op(objective, typeof.((u, problem, quad))...), NaN)
    end
end

# run a function with GC disabled (workaround for Enzyme/GC bug that results in segfaults)
function withoutgc(f::F, args::Vararg{Any,N}; kwargs...) where {F,N}
    GC.gc(false)  # minimize OOM risk
    gcstate = GC.enable(false)
    try
        return f(args...; kwargs...)
    finally
        GC.enable(gcstate)
    end
end

### objective gradient
# function make_dobjective(op::Orderparameters)
#     dop = make_zero(op)
#     return function dobjective!(du, u, problem, quad, op)
#         make_zero!(du)
#         make_zero!(dop)
#         withoutgc(
#             autodiff,
#             Reverse,
#             objective,
#             Active,
#             Duplicated(u, du),
#             Const(problem),
#             Const(quad),
#             Duplicated(op, dop),
#         )
#         return nothing
#     end
# end

function dobjective!(du, u, problem, quad)
    make_zero!(du)
    withoutgc(
        autodiff,
        Reverse,
        objective,
        Active,
        Duplicated(u, du),
        Const(problem),
        Const(quad),
    )
    return nothing
end

# function make_dnanobjective(op::Orderparameters)
#     dop = make_zero(op)
#     return function dobjective!(du, u, problem, quad, op, warn...)
#         make_zero!(du)
#         make_zero!(dop)
#         try
#             withoutgc(
#                 autodiff,
#                 Reverse,
#                 objective,
#                 Active,
#                 Duplicated(u, du),
#                 Const(problem),
#                 Const(quad),
#                 Duplicated(op, dop),
#             )
#         catch e
#             checkRSBexceptions(e, warn...; actionmsg="Returning NaN in gradient") || rethrow()
#             du .= NaN
#         end
#         return nothing
#     end
# end

function dnanobjective!(du, u, problem, quad, warn...)
    try
        dobjective!(du, u, problem, quad)
    catch e
        checkRSBexceptions(e, warn...; actionmsg="Returning NaN in gradient") || rethrow()
        du .= NaN
    end
    return nothing
end

## optimizer
function optimize(
    problem::AbstractDisorderProblem{T,inf}, quad::AbstractQuadrature{T}, op0=initorderparameters(problem);
    warn_on_handled_exceptions::Bool=false, callback=nothing, max_wall_time=60.0,
    tol=eps(T)^(11 // 12), acceptable_tol=tol, limited_memory_max_history=2^16,
    ipoptoptions...,
) where {T,inf}
    # op = orderparameters(problem, u0)
    # dnanobjective! = make_dnanobjective(op)

    # wrap objective function and gradient
    # eval_f(u) = nanobjective(u, problem, quad, op, warn_on_handled_exceptions)
    # eval_grad_f(u, du) = dnanobjective!(du, u, problem, quad, op, warn_on_handled_exceptions)
    eval_f(u) = nanobjective(u, problem, quad, warn_on_handled_exceptions)
    eval_grad_f(u, du) = dnanobjective!(du, u, problem, quad, warn_on_handled_exceptions)

    # wrap constraint function and jacobian
    eval_g(u, new_g) = constraintf!(new_g, u, problem)
    eval_jac_g(u, _, _, values) = dconstraintf!(values, u, problem)
    function eval_jac_g(_, rows, cols, ::Nothing)
        # populate sparsity structure
        fill!(rows, 1)
        copyto!(cols, 1:length(cols))
        return nothing
    end

    # get initial state, bounds, and constraint values
    u0 = state(problem, op0)
    u_L, u_U = statebounds(u0, problem)
    g_L, g_U = constraintvalues(problem)
    m = length(g_L)
    nele_jac = dconstraintlength(problem)

    # create and initialize Ipopt problem
    ipoptprob = CreateIpoptProblem(
        length(u0),   # problem dimension
        u_L,          # lower bounds
        u_U,          # upper bounds
        m,            # constraint function dimension
        g_L,          # constraint function lower bounds
        g_U,          # constraint function upper bounds
        nele_jac,     # number of nonzero elements in constraint function jacobian
        0,            # number of nonzero elements in hessian of lagrangian
        eval_f,       # objective function
        eval_g,       # constraint function
        eval_grad_f,  # gradient of objective function
        eval_jac_g,   # jacobian of constraint function
        nothing,      # hessian of lagrangian
    )
    ipoptprob.x .= u0

    # fixed options
    if dconstraintfisconstant(problem)
        addipoptoption(ipoptprob, "jac_d_constant", "yes")
    end
    addipoptoption(ipoptprob, "hessian_approximation", "limited-memory")
    addipoptoption(ipoptprob, "limited_memory_initialization", "scalar2")
    addipoptoption(ipoptprob, "bound_relax_factor", zero(T))
    addipoptoption(ipoptprob, "bound_push", sqrt(eps(T)))
    addipoptoption(ipoptprob, "bound_frac", sqrt(eps(T)))
    addipoptoption(ipoptprob, "slack_bound_push", sqrt(eps(T)))
    addipoptoption(ipoptprob, "slack_bound_frac", sqrt(eps(T)))
    # addipoptoption(ipoptprob, "nlp_scaling_method", "none")
    # addipoptoption(ipoptprob, "mu_init", sqrt(eps(T)))
    # addipoptoption(ipoptprob, "mu_strategy", "adaptive")
    # addipoptoption(ipoptprob, "mu_linear_decrease_factor", 0.5)
    # addipoptoption(ipoptprob, "mu_superlinear_decrease_power", nextfloat(1.0))
    # addipoptoption(ipoptprob, "barrier_tol_factor", 2.0^(-12))

    # configurable options with defaults
    addipoptoption(ipoptprob, "max_wall_time", max_wall_time)
    addipoptoption(ipoptprob, "limited_memory_max_history", limited_memory_max_history)

    # user-provided options
    foreach(option -> addipoptoption(ipoptprob, option...), ipoptoptions)

    if !isnothing(callback)
        setipoptcallback(ipoptprob, callback)
    end

    # tolerances
    addipoptoption(ipoptprob, "tol", T(tol))
    addipoptoption(ipoptprob, "acceptable_tol", T(acceptable_tol))

    addipoptoption(ipoptprob, "constr_viol_tol", T(sqrt(tol)))
    addipoptoption(ipoptprob, "compl_inf_tol", T(sqrt(tol)))
    addipoptoption(ipoptprob, "acceptable_constr_viol_tol", T(sqrt(acceptable_tol)))
    addipoptoption(ipoptprob, "acceptable_compl_inf_tol", T(sqrt(acceptable_tol)))

    # for tiny_step_tol we assume that any nominally unbounded variables are scaled or
    # constrained to nevertheless vary over a similar scale as the bounded variables, hence:
    addipoptoption(ipoptprob, "tiny_step_tol", eps(maximum(filter(isfinite, u_U - u_L))))
    addipoptoption(ipoptprob, "tiny_step_y_tol", T(fourthroot(tol)))

    # solve and check
    IpoptSolve(ipoptprob)
    if ipoptprob.status == 1
        @warn "Optimization converged to acceptable but not desired tolerance"
    elseif ipoptprob.status != 0
        # @error
        error("Optimization did not converge, returned code $(ipoptprob.status)")
    end

    # return minimizer
    return ipoptprob.x
end

# TODO: if this is to be kept, we need a remake function so we don't have to specialize on
# the problem type
# function optimize_sequence(
#     problem::SKProblem{T}, quad::AbstractQuadrature{T}; kwargs...
# ) where {T}
#     (; k, β, h) = problem
#     u_l = initstate(SKProblem(1, β, h))
#     for l in 2:k
#         problem_l = SKProblem(l, β, h)
#         u_l = augment_state(u_l)
#         u_l = optimize(problem_l, quad, u_l; kwargs...)
#         @info "Optimization converged for k = $l"
#     end
#     return u_l
# end
