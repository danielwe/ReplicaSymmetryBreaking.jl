### unpack captured/composite exceptions
unpackexceptions(@nospecialize(e::Exception)) = Exception[e]

function unpackexceptions(e::CompositeException)
    return mapreduce(unpackexceptions, vcat, e)
end

function unpackexceptions(e::TaskFailedException)
    _unpackexceptions(stackex) = unpackexceptions(stackex.exception)
    return mapreduce(_unpackexceptions, vcat, current_exceptions(e.task))
end

EnzymeRules.inactive(::typeof(unpackexceptions), args...) = true

### concatenate error messages from composite exceptions
function errormessage(e::Exception, args...)
    return errormessage(unpackexceptions(e), args...)
end

function errormessage(stack, delim="\n")
    return join(Iterators.map(ex -> sprint(showerror, ex), stack), delim)
end

EnzymeRules.inactive(::typeof(errormessage), args...) = true

### check known exceptions
function checkRSBexceptions(e::Exception, warn=true; actionmsg=nothing)
    unpacked = unpackexceptions(e)
    for ex in unpacked
        isa(ex, AbstractRSBException) || return false
    end
    if warn
        actionmsgfull = isnothing(actionmsg) ? "" : "\n$actionmsg"
        exmsgs = errormessage(unpacked, "\n- ")
        @warn "Exceptions encountered while computing RSB free energy:\n- $exmsgs$actionmsgfull"
    end
    return true
end

EnzymeRules.inactive(::typeof(checkRSBexceptions), args...) = true
