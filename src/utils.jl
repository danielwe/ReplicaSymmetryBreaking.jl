# Unpack and handle captured/composite exceptions
unpackexceptions(@nospecialize(e::Exception)) = Exception[e]

function unpackexceptions(e::CompositeException)
    return mapreduce(unpackexceptions, vcat, e)
end

function unpackexceptions(e::TaskFailedException)
    _unpackexceptions(stackex) = unpackexceptions(stackex.exception)
    return mapreduce(_unpackexceptions, vcat, current_exceptions(e.task))
end

function errormessage(e::Exception, args...)
    return errormessage(unpackexceptions(e), args...)
end

function errormessage(stack, delim="\n")
    return join(Iterators.map(ex -> sprint(showerror, ex), stack), delim)
end

function handleexceptions(e::Exception, ::Type{KnownEx}; warn=true) where {KnownEx}
    e_unpacked = unpackexceptions(e)
    for ex in e_unpacked
        isa(ex, KnownEx) || return false
    end
    if warn
        msg = errormessage(e_unpacked, "\n- ")
        @warn "Handling the following exceptions:\n- " * msg
    end
    return true
end

EnzymeRules.inactive(::typeof(unpackexceptions), args...) = true
EnzymeRules.inactive(::typeof(errormessage), args...) = true
EnzymeRules.inactive(::typeof(handleexceptions), args...) = true

# Use multiple dispatch to streamline Ipopt option setting
addipoptoption(p, name, value) = AddIpoptStrOption(p, string(name), value)
addipoptoption(p, name, value::Integer) = AddIpoptIntOption(p, string(name), value)
addipoptoption(p, name, value::AbstractFloat) = AddIpoptNumOption(p, string(name), value)

# Set Ipopt callback using an Optimization.jl-compatible callback function
function setipoptcallback(p, callback)
    function intermediate_callback(
        alg_mod::Cint,
        iter_count::Cint,
        obj_value::Float64,
        inf_pr::Float64,
        inf_du::Float64,
        mu::Float64,
        d_norm::Float64,
        regularization_size::Float64,
        alpha_du::Float64,
        alpha_pr::Float64,
        ls_trials::Cint,
    )
        state = (;
            iter=iter_count,
            u=p.x,
            objective=obj_value,
            gradient=nothing,
            hessian=nothing,
            original=(;
                alg_mod,
                iter_count,
                obj_value,
                inf_pr,
                inf_du,
                mu,
                d_norm,
                regularization_size,
                alpha_du,
                alpha_pr,
                ls_trials,
            ),
        )
        return !callback(state, obj_value)
    end
    SetIntermediateCallback(p, intermediate_callback)
    return nothing
end
