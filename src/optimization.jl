## Workspace containing order parameters and preallocated buffers
# struct Workspace{T,O<:Orderparameters{T},B<:Tracebuffers{T}}
#     op::O
#     buffers::B
# end
#
# function Workspace(
#     problem::AbstractRSBProblem{T}, quad::AbstractRSBQuadrature{T}; ntasks
# ) where {T}
#     op = Orderparameters(problem)
#     buffers = Tracebuffers(quad; ntasks)
#     return Workspace{T,typeof(op),typeof(buffers)}(op, buffers)
# end

## Objective
function objective(
    u::AbstractVector{T},
    problem::AbstractRSBProblem{T},
    quad::AbstractRSBQuadrature{T},
    op::Orderparameters{T},
    # (; op, buffers)::Workspace{T},
) where {T}
    updateorderparameters!(op, problem, u; check=true, checksums=false)
    F = freeenergy(problem, quad, op)  #, buffers)
    return -F
end

# function nanobjective(u, problem, quad, ws, warn...)
function nanobjective(u, problem, quad, op, warn...)
    try
        # return objective(u, problem, quad, )
        return objective(u, problem, quad, op)
    catch e
        # Do not handle Gaussian leak errors, they require parameter changes
        # KnownEx = Union{GaussKronrodPrecisionError,OrderparametersError}
        handleexceptions(e, AbstractRSBException, warn...) || rethrow()
        # return convert(Base.promote_op(objective, typeof.((u, problem, quad, ws))...), NaN)
        return convert(Base.promote_op(objective, typeof.((u, problem, quad, op))...), NaN)
    end
end

@inline function withoutgc(f::F, args::Vararg{Any,N}) where {F,N}  # run f with GC disabled
    GC.gc(true)  # minimize OOM risk
    gcstate = GC.enable(false)
    try
        return f(args...)
    finally
        GC.enable(gcstate)
    end
end

## Objective gradient
# function make_dobjective(ws::Workspace)
function make_dobjective(op::Orderparameters)
    # dws = make_zero(ws)
    dop = make_zero(op)
    # return function dobjective!(du, u, problem, quad, ws)
    return function dobjective!(du, u, problem, quad, op)
        make_zero!(du)
        # make_zero!(dws)
        make_zero!(dop)
        withoutgc(
            autodiff,
            Reverse,
            objective,
            Active,
            Duplicated(u, du),
            Const(problem),
            Const(quad),
            # Duplicated(ws, dws),
            Duplicated(op, dop),
        )
        return nothing
    end
    # return function dobjective!(du, u, problem, quad, op)
    #     f(u_) = objective(u_, problem, quad, op)
    #     du .= grad(central_fdm(2, 1; factor=getrtol(quad) / eps(eltype(u))), f, u)[1]
    #     return nothing
    # end
end

function make_dnanobjective(op::Orderparameters)
    # dws = make_zero(ws)
    dop = make_zero(op)
    # return function dobjective!(du, u, problem, quad, ws, warn...)
    return function dobjective!(du, u, problem, quad, op, warn...)
        make_zero!(du)
        # make_zero!(dws)
        make_zero!(dop)
        try
            withoutgc(
                autodiff,
                Reverse,
                objective,
                Active,
                Duplicated(u, du),
                Const(problem),
                Const(quad),
                # Duplicated(ws, dws),
                Duplicated(op, dop),
                Const.(warn)...,
            )
        catch e
            handleexceptions(e, AbstractRSBException, warn...) || rethrow()
            du .= NaN
        end
        return nothing
    end
    # return function dobjective!(du, u, problem, quad, op, warn...)
    #     f(u_) = nanobjective(u_, problem, quad, op, warn...)
    #     du .= grad(central_fdm(2, 1; factor=getrtol(quad) / eps(eltype(u))), f, u)[1]
    #     return nothing
    # end
end

## Solver
function optimize(
    problem::AbstractRSBProblem{T,inf}, quad::AbstractRSBQuadrature{T}, u0=initstate(problem);
    # ntasks=Threads.threadpoolsize(),
    ntolsteps=1,
    warn_on_handled_exceptions=false, callback=nothing, max_wall_time=60.0,
    tol=eps(T)^(11 // 12), acceptable_tol=tol, limited_memory_max_history=2^16,
    ipoptoptions...,
) where {T,inf}
    # Allocate workspace and create function and gradient closures
    # ws = Workspace(problem, quad; ntasks)

    op = Orderparameters(problem)
    eval_f(u) = nanobjective(u, problem, quad, op, warn_on_handled_exceptions)
    # eval_f(u) = objective(u, problem, quad, op)

    dnanobjective! = make_dnanobjective(op)  # should we skip the nan version here?
    # dobjective! = make_dobjective(op)
    # eval_grad_f(u, du) = dobjective!(du, u, problem, quad, ws, warn_on_handled_exceptions)
    function eval_grad_f(u, du)
        # dnanobjective!(du, u, problem, quad, op, warn_on_handled_exceptions)
        dnanobjective!(du, u, problem, quad, op)
        # any(isnan, du) && @warn "Got NaN in gradient"
        # @info """u = $u
        #     du = $du"""
        return nothing
    end

    (; k, β) = problem
    βtransformed = applyinverse(atransform, β)
    u_L = fill(zero(T), length(u0))
    # u_U = fill(oneunit(T) / 2, length(u0))
    u_U = [fill(T(Inf), k); fill(oneunit(T) / 2, length(u0) - k)]
    # u_U = fill(T(Inf), length(u0))
    # u_U = [fill(T(βtransformed), k); fill(oneunit(T) / 2, length(u0) - k)]

    if βtransformed == Inf  # no constraints
        m = 0
        nele_jac = 0
        g_L = T[]
        g_U = T[]
        eval_g = (@nospecialize(args...)) -> nothing
        eval_jac_g = (@nospecialize(args...)) -> nothing
        # m = 1
        # nele_jac = length(u0) - k
        # g_L = T[1]
        # g_U = T[1]
        # eval_g = function (u, new_g)
        #     Base.require_one_based_indexing(u, new_g)
        #     new_g[1] = sum(abs2, @view u[(k + 1):end])
        #     # new_g[1] = sum(@view u[(k + 1):end])
        #     return nothing
        # end
        # eval_jac_g = function (u, rows, cols, values)
        #     Base.require_one_based_indexing(u)
        #     @views if isnothing(values)
        #         # populate sparsity structure
        #         fill!(rows, 1)
        #         cols .= k .+ (1:length(cols))
        #     else
        #         # populate Jacobian values
        #         values .= 2 .* u[(k + 1):end]
        #         # fill!(values, oneunit(eltype(values)))
        #     end
        #     return nothing
        # end
    else
        m = 1
        nele_jac = k
        g_L = T[-Inf]  # setting g_L would be redundant given u_L
        g_U = T[βtransformed]
        eval_g = function (u, new_g)
            Base.require_one_based_indexing(u, new_g)
            # new_g[1] = sum(@view u[1:k])
            new_g[1] = foldl(+, @view u[1:k])
            # new_g[1] = foldr(+, @view u[1:k])
            # new_g[1] = mapfoldr(atransform, +, @view u[1:k])
            return nothing
        end
        eval_jac_g = function (_, rows, cols, values)
            if isnothing(values)
                # populate sparsity structure
                fill!(rows, 1)
                copyto!(cols, 1:length(cols))
            else
                # populate Jacobian values
                fill!(values, oneunit(eltype(values)))
                # map!(getjacobian(atransform), values, @view u[1:k])
            end
            return nothing
        end
        # m = 2
        # nele_jac = length(u0)
        # g_L = T[-Inf, 1]  # setting g_L would be redundant given u_L
        # g_U = T[βtransformed, 1]
        # eval_g = function (u, new_g)
        #     Base.require_one_based_indexing(u, new_g)
        #     # new_g[1] = sum(@view u[1:k])
        #     new_g[1] = foldl(+, @view u[1:k])
        #     # new_g[1] = foldr(+, @view u[1:k])
        #     # new_g[1] = mapfoldr(atransform, +, @view u[1:k])
        #     new_g[2] = sum(abs2, @view u[(k + 1):end])
        #     # new_g[2] = sum(@view u[(k + 1):end])
        #     return nothing
        # end
        # eval_jac_g = function (u, rows, cols, values)
        #     Base.require_one_based_indexing(u, rows)
        #     @views if isnothing(values)
        #         # populate sparsity structure
        #         fill!(rows[1:k], 1)
        #         fill!(rows[(k + 1):end], 2)
        #         copyto!(cols, 1:length(cols))
        #     else
        #         Base.require_one_based_indexing(values)
        #         # populate Jacobian values
        #         fill!(values[1:k], oneunit(eltype(values)))
        #         values[(k + 1):end] .= 2 .* u[(k + 1):end]
        #         # fill!(values, oneunit(eltype(values)))
        #     end
        #     return nothing
        # end
    end

    # Create and initialize Ipopt problem
    prob = CreateIpoptProblem(
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
    # prob = CreateIpoptProblem(
    #     length(u0),         # problem dimension
    #     zero(u0),           # lower bounds
    #     oneunit.(u0) ./ 2,  # upper bounds
    #     0,                  # constraint function dimension
    #     T[],                # constraint function lower bounds
    #     T[],                # constraint function upper bounds
    #     0,                  # number of nonzero elements in constraint function jacobian
    #     0,                  # number of nonzero elements in hessian of lagrangian
    #     eval_f,             # objective function
    #     (@nospecialize(args...)) -> nothing,  # constraint function
    #     eval_grad_f,        # gradient of objective function
    #     (@nospecialize(args...)) -> nothing,  # jacobian of constraint function
    #     nothing,            # hessian of lagrangian
    # )
    # prob.x .= u0
    # @info """initial state set
    #     u0 = $(prob.x)"""

    # Fixed options
    # addipoptoption(prob, "jac_c_constant", "yes")
    addipoptoption(prob, "jac_d_constant", "yes")
    addipoptoption(prob, "hessian_approximation", "limited-memory")
    addipoptoption(prob, "limited_memory_initialization", "scalar2")
    addipoptoption(prob, "bound_relax_factor", zero(T))
    # addipoptoption(prob, "nlp_scaling_method", "none")
    # addipoptoption(prob, "mu_init", sqrt(eps(T)))
    # addipoptoption(prob, "mu_strategy", "adaptive")
    # addipoptoption(prob, "adaptive_mu_restore_previous_iterate", "yes")
    # addipoptoption(prob, "mu_allow_fast_monotone_decrease", "no")
    # addipoptoption(prob, "mu_linear_decrease_factor", 0.5)
    # addipoptoption(prob, "mu_superlinear_decrease_power", nextfloat(1.0))
    # addipoptoption(prob, "barrier_tol_factor", 2.0^(-12))
    # addipoptoption(prob, "alpha_red_factor", 0.05)
    addipoptoption(prob, "tiny_step_tol", 2eps(T))
    addipoptoption(prob, "bound_push", eps(T) / 2)
    addipoptoption(prob, "bound_frac", eps(T))
    addipoptoption(prob, "slack_bound_push", eps(T) / 2)
    addipoptoption(prob, "slack_bound_frac", eps(T))

    # Default options
    addipoptoption(prob, "max_wall_time", max_wall_time)
    addipoptoption(prob, "limited_memory_max_history", limited_memory_max_history)

    # User-provided options
    foreach(option -> addipoptoption(prob, option...), ipoptoptions)

    if !isnothing(callback)
        setipoptcallback(prob, callback)
    end

    u_ = copy(u0)

    # Solve at looser tolerance first, then tighten
    # This is to compensate for Ipopt's overly aggressive lowering of the barrier parameter
    for i in 1:ntolsteps
        (acceptable_tol_, tol_) = (acceptable_tol, tol) .^ (i // ntolsteps)
        addipoptoption(prob, "tol", T(tol_))
        addipoptoption(prob, "acceptable_tol", T(acceptable_tol_))

        addipoptoption(prob, "constr_viol_tol", T(sqrt(tol_)))
        addipoptoption(prob, "compl_inf_tol", T(sqrt(tol_)))
        addipoptoption(prob, "acceptable_constr_viol_tol", T(sqrt(acceptable_tol_)))
        addipoptoption(prob, "acceptable_compl_inf_tol", T(sqrt(acceptable_tol_)))

        addipoptoption(prob, "tiny_step_y_tol", T(fourthroot(tol_)))

        if i > 1
            addipoptoption(prob, "warm_start_init_point", "yes")
        end

        # Solve and verify at this tolerance
        prob.x .= u_
        # try
        IpoptSolve(prob)
        # catch e
        #     @error "" exception = (e, catch_backtrace())
        #     break
        # finally
        u_ .= prob.x
        # end
        if prob.status == 1
            @warn "Optimization converged to acceptable but not desired tolerance"
        elseif prob.status != 0
            error("Optimization did not converge, returned code $(prob.status)")
            # @error "Optimization did not converge, returned code $(prob.status)"
            # break
        end
    end
    # Return minimizer
    return u_
end

function optimize_sequence(
    problem::SKProblem{T}, quad::AbstractRSBQuadrature{T};
    tol=eps(T)^(11 // 12), acceptable_tol=tol, kwargs...,
) where {T}
    (; k, β, h) = problem
    u_l = initstate(SKProblem(1, β, h))
    for l in 2:k
        problem_l = SKProblem(l, β, h)
        u_l = augment_state(u_l)
        tol_, acceptable_tol_ = tol, acceptable_tol
        # tol_, acceptable_tol_ = (tol, acceptable_tol) .^ ((k + l) // (2k))
        # tol_, acceptable_tol_ = (l < k) ? sqrt.((tol, acceptable_tol)) : (tol, acceptable_tol)
        u_l = optimize(
            problem_l, quad, u_l; tol=tol_, acceptable_tol=acceptable_tol_, kwargs...
        )
        @info "Optimization converged for k = $l"
        # @show u_l
        # @show Orderparameters(problem_l, u_l)
    end
    return u_l
end

## Objective Hessian-vector product
#
# TODO: try this with the new rational approximation, which seems to have much less autodiff
# overhead by virtue of not importing a ChainRules rule (but the primal is costly, so this
# coule still be slow)
#
# Not efficient enough to be used in the optimization, and doesn't work with multithreading,
# but possibly useful for investiating Hessian sparsity/rank for the given parametrization
# Call Enzyme.API.looseTypeAnalysis!(true) before use
# function make_hvpobjective(uprototype::AbstractVector{T}, ws::Workspace{T}) where {T}
# function make_hvpobjective(uprototype::AbstractVector{T}, op::Orderparameters{T}) where {T}
#     function _dobjective!(du, u, problem, quad, op, dop=make_zero(op))
#         make_zero!(du)
#         autodiff(
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
#     du = make_zero(uprototype)
#     bop = make_zero(op)
#     dop = make_zero(op)
#     bdop = make_zero(dop)
#     return function hvpobjective!(Hv, v, u, problem, quad, op)
#         make_zero!(bop)
#         make_zero!(dop)
#         make_zero!(bdop)
#         make_zero!(Hv)
#         withoutgc(
#             autodiff,
#             Forward,
#             _dobjective!,
#             Const,
#             Duplicated(du, Hv),
#             Duplicated(u, v),
#             Const(problem),
#             Const(quad),
#             Duplicated(op, bop),
#             Duplicated(dop, bdop),
#         )
#         return nothing
#     end
# end
