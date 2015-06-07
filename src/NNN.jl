module NNN

using InplaceOps
using Base.LinAlg.BLAS

include("type.jl")
include("backprop.jl")
include("activation.jl")
include("filter.jl")
include("cost.jl")
include("optimize.jl")
include("util.jl")

export Layer,
       HiddenLayer,
       OutputLayer,
       RecurrentLayer,
       Network,
       Optimizer,
       cross_entropy_cost,
       rnn_cross_entropy_cost,
       quadratic_cost,
       rnn_quadratic_cost,
       report,
       train,
       gradient_clip!,
       Filter,
       Dropout

end
