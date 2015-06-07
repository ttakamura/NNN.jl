immutable FilterContext
    time_step::Int64
end

function forward!(filters::Vector{Filter}, l::Layer; time_step=0)
    context = FilterContext(time_step)
    for i in 1:length(filters)
        if filters[i].enabled
            forward!(filters[i], l, context)
        end
    end
end

function backprop!(filters::Vector{Filter}, l::Layer; time_step=0)
    context = FilterContext(time_step)
    for i in 1:length(filters)
        if filters[i].enabled
            backprop!(filters[i], l, context)
        end
    end
end

function forward!(f::Filter, l::Layer, context::FilterContext)
    # No-op
end

function backprop!(f::Filter, l::Layer, context::FilterContext)
    # No-op
end

# ----------------------------------------------------------------
immutable Dropout <: Filter
    enabled::Bool
    drop::Float32
    scale::Float32
    mask::Signal
    Dropout(drop) = new(true, drop, 1.0/(1.0 - drop))
end

function forward!(drop::Dropout, l::Layer, context::FilterContext)
    mask = (rand(size(l.Z)) .> drop.drop)
    copy!(drop.mask, mask)
    drop!(drop, l.Z)
end

function backprop!(drop::Dropout, l::Layer, context::FilterContext)
    drop!(drop, l.ΔE)
end

function forward!(drop::Dropout, l::RecurrentLayer, context::FilterContext)
    error("Not yet implemented")
end

function backprop!(drop::Dropout, l::RecurrentLayer, context::FilterContext)
    error("Not yet implemented")
end

function drop!(drop::Dropout, X::Signal)
    for col=1:size(X,2), row=1:size(X,1)
        X[row, col] = X[row, col] * drop.scale * drop.mask[row, col]
    end
end

# ----------------------------------------------------------------
immutable GradientClip <: Filter
    enabled::Bool
    threshold::Float32
    GradientClip(threshold) = new(true, threshold)
end

function backprop!(grad::GradientClip, l::Layer, context::FilterContext)
    gradient_clip!(l.ΔE, grad.threshold)
end

function backprop!(grad::GradientClip, l::RecurrentLayer, context::FilterContext)
    t = context.time_step
    gradient_clip!(sub(l.ΔE,:,t:t),  grad.threshold)
end

function gradient_clip!(g, limit::Float32)
    nm = norm(g,1)
    if nm > limit
        rate = limit / nm
        ng = @in1! g .* rate
    end
    g
end
