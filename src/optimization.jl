## Workspace containing order parameters and preallocated buffers
struct Workspace{T,ntasks}
    op::Orderparameters{T}
    buffers::Tracebuffers{T,ntasks}
end

function Workspace(
    u::AbstractVector{T}, problem::AbstractRSBProblem{T}, quad::GaussKronrod{T}; ntasks
) where {T}
    op = Orderparameters(problem, u)
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
    return freeenergy(problem, quad, op, buffers)
end

function nanobjective(args...)
    try
        return objective(args...)
    catch e
        handleexceptions(e, AbstractRSBException) || rethrow()
        return convert(Base.promote_op(objective, typeof.(args)...), NaN)
    end
end

## Objective gradient
function make_dobjective(ws::Workspace, objective::O=objective) where {O}
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

make_nandobjective(ws) = make_dobjective(ws, nanobjective)

## Solver
function optimize(
    problem::AbstractRSBProblem{T}, u0::AbstractVector{T}=initstate(problem);
    quadn, quadcutoff=T(Inf), quadscale=8one(T), quadtol=16eps(T), quadstretchmode=nothing,
    kwargs...,
) where {T}
    quadkws = if isnothing(quadstretchmode)
        (; scale=T(quadscale), tol=T(quadtol))
    else
        (; scale=T(quadscale), tol=T(quadtol), stretchmode=quadstretchmode)
    end
    quad = GaussKronrod(quadn, T(quadcutoff); quadkws...)
    return _optimize(u0, problem, quad; kwargs...)
end

function _optimize(
    u0::AbstractVector{T}, problem::AbstractRSBProblem{T}, quad::GaussKronrod{T};
    optreltol=256eps(T), ntasks=Threads.threadpoolsize(), callback=nothing, algoptions...,
) where {T}
    ws = Workspace(u0, problem, quad; ntasks)
    F = let problem=problem, quad=quad, ws=ws
        f = if isnothing(callback)
            f_nocallback(u) = -nanobjective(u, problem, quad, ws)
        else
            let callback=callback
                function f_callback!(u)
                    obj = -nanobjective(u, problem, quad, ws)
                    if callback((; u), obj)
                        # Can't find a way to stop Nonconvex.jl
                    end
                    return obj
                end
            end
        end
        g = let du=similar(u0), dobjective! = make_nandobjective(ws)
            function g(u)
                dobjective!(du, u, problem, quad, ws)
                @. du = -du
                return du
            end
        end
        CustomGradFunction(f, g)
    end
    model = Model(F)
    addvar!(model, zero(u0), oneunit.(u0) ./ 2)
    alg = IpoptAlg()
    options = IpoptOptions(;
        first_order=true, tol=T(optreltol), acceptable_tol=T(optreltol),
        linear_constraints=true, algoptions...,
    )
    result = Nonconvex.optimize(model, alg, u0; options)
    if result.status == 1
        @warn "Optimization converged to acceptable but not desired tolerance"
    elseif result.status != 0
        error("Optimization did not converge, returned code $(result.status)")
    end
    return result.minimizer
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
