### Configure Ipopt problems
addipoptoption(p, name, value) = AddIpoptStrOption(p, string(name), value)
addipoptoption(p, name, value::Integer) = AddIpoptIntOption(p, string(name), value)
addipoptoption(p, name, value::AbstractFloat) = AddIpoptNumOption(p, string(name), value)

# Use an Optimization.jl-compatible callback with Ipopt
function setipoptcallback(p, @nospecialize(callback))
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

