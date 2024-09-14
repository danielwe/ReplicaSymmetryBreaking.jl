## Workspace containing order parameters and preallocated buffers
struct Workspace{T,ntasks}
    op::Orderparameters{T}
    buffers::Tracebuffers{T,ntasks}
end

function Workspace(problem::AbstractRSBProblem{T}, quad::GaussKronrod{T}; ntasks) where {T}
    op = Orderparameters(problem)
    buffers = Tracebuffers(T, length(quad.weights); ntasks)
    return Workspace{T,ntasks}(op, buffers)
end

function _make_zero!(ws::Workspace)
    _make_zero!(ws.op)
    _make_zero!(ws.buffers)
    return nothing
end

## Objective
function objective(
    u::AbstractVector{T},
    problem::AbstractRSBProblem{T},
    quad::GaussKronrod{T},
    (; op, buffers)::Workspace{T},
) where {T}
    updateorderparameters!(op, problem, u)
    return -freeenergy(problem, quad, op, buffers)
end

function nanobjective(u, problem, quad, ws, warn...)
    try
        return objective(u, problem, quad, ws)
    catch e
        # Do not handle Gaussian leak errors, they require parameter changes
        KnownEx = Union{<:GaussKronrodPrecisionError,<:OrderparametersError}
        handleexceptions(e, KnownEx, warn...) || rethrow()
        return convert(Base.promote_op(objective, typeof.((u, problem, quad, ws))...), NaN)
    end
end

## Objective gradient
function make_dobjective(ws::Workspace)
    dws = make_zero(ws)
    return function dobjective!(du, u, problem, quad, ws)
        _make_zero!(du)
        _make_zero!(dws)
        autodiff(
            Reverse,
            objective,
            Active,
            Duplicated(u, du),
            Const(problem),
            Const(quad),
            Duplicated(ws, dws),
        )
        return nothing
    end
end

function make_dnanobjective(ws::Workspace)
    dws = make_zero(ws)
    return function dnanobjective!(du, u, problem, quad, ws, warn...)
        _make_zero!(du)
        _make_zero!(dws)
        autodiff(
            Reverse,
            nanobjective,
            Active,
            Duplicated(u, du),
            Const(problem),
            Const(quad),
            Duplicated(ws, dws),
            Const.(warn)...,
        )
        return nothing
    end
end

## Solver
function optimize(
    problem::AbstractRSBProblem{T}, quad::GaussKronrod{T}, u0=initstate(problem);
    ntasks=Threads.threadpoolsize(), warn_on_handled_exceptions=false, callback=nothing,
    max_wall_time=60.0, tol=8eps(T), acceptable_tol=tol, limited_memory_max_history=256,
    ipoptoptions...,
) where {T}
    # Allocate workspace and create function and gradient closures
    ws = Workspace(problem, quad; ntasks)
    eval_f(u) = nanobjective(u, problem, quad, ws, warn_on_handled_exceptions)

    dnanobjective! = make_dnanobjective(ws)
    eval_grad_f(u, du) = dnanobjective!(du, u, problem, quad, ws, warn_on_handled_exceptions)

    # Create and initialize Ipopt problem
    prob = CreateIpoptProblem(
        length(u0),         # problem dimension
        zero(u0),           # lower bounds
        oneunit.(u0) ./ 2,  # upper bounds
        0,                  # constraint function dimension
        T[],                # constraint function lower bounds
        T[],                # constraint function upper bounds
        0,                  # number of nonzero elements in constraint function jacobian
        0,                  # number of nonzero elements in hessian of lagrangian
        eval_f,             # objective function
        (_...) -> nothing,  # constraint function
        eval_grad_f,        # gradient of objective function
        (_...) -> nothing,  # jacobian of constraint function
        nothing,            # hessian of lagrangian
    )
    prob.x .= u0

    # Fixed options
    addipoptoption(prob, "hessian_approximation", "limited-memory")

    # Default options
    addipoptoption(prob, "max_wall_time", max_wall_time)
    addipoptoption(prob, "tol", T(tol))
    addipoptoption(prob, "acceptable_tol", T(acceptable_tol))
    addipoptoption(prob, "limited_memory_max_history", limited_memory_max_history)

    # User-provided options
    foreach(option -> addipoptoption(prob, option...), ipoptoptions)

    if !isnothing(callback)
        setipoptcallback(prob, callback)
    end

    # Solve problem, inspect result, and return minimzer
    IpoptSolve(prob)
    if prob.status == 1
        @warn "Optimization converged to acceptable but not desired tolerance"
    elseif prob.status != 0
        error("Optimization did not converge, returned code $(prob.status)")
    end
    return prob.x
end

## Objective Hessian-vector product
# Not efficient enough to be used in the optimization, and doesn't work with multithreading,
# but possibly useful for investiating Hessian sparsity/rank for the given parametrization
# Call Enzyme.API.looseTypeAnalysis!(true) before use
function make_hvpobjective(uprototype::AbstractVector{T}, ws::Workspace{T}) where {T}
    function _dobjective_deferred!(du, u, problem, quad, ws, dws=make_zero(ws))
        _make_zero!(du)
        autodiff_deferred(
            Reverse,
            objective,
            Active,
            Duplicated(u, du),
            Const(problem),
            Const(quad),
            Duplicated(ws, dws),
        )
        return nothing
    end
    du = make_zero(uprototype)
    bws = make_zero(ws)
    dws = make_zero(ws)
    bdws = make_zero(dws)
    return function hvpobjective!(Hv, v, u, problem, quad, ws)
        _make_zero!(bws)
        _make_zero!(dws)
        _make_zero!(bdws)
        _make_zero!(Hv)
        autodiff(
            Forward,
            _dobjective_deferred!,
            Const,
            Duplicated(du, Hv),
            Duplicated(u, v),
            Const(problem),
            Const(quad),
            Duplicated(ws, bws),
            Duplicated(dws, bdws),
        )
        return nothing
    end
end
