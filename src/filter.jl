function forward!(filters::Vector{Filter}, l::Layer)
    for i in 1:length(filters)
        if filters[i].enabled
            forward!(filters[i], l)
        end
    end
end

function backprop!(filters::Vector{Filter}, l::Layer)
    for i in 1:length(filters)
        if filters[i].enabled
            backprop!(filters[i], l)
        end
    end
end

# ----------------------------------------------------------------
immutable Dropout <: Filter
    enabled::Bool
    drop::Float32
    scale::Float32
    mask::Signal
    Dropout(drop) = new(true, drop, 1.0/(1.0 - drop))
end

function forward!(drop::Dropout, l::Layer)
    mask = (rand(size(l.Z)) .> drop.drop)
    copy!(drop.mask, mask)
    drop!(drop, l.Z)
end

function backprop!(drop::Dropout, l::Layer)
    drop!(drop, l.ΔE)
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

function forward!(grad::GradientClip, l::Layer)
    # NO-OP
end

function backprop!(grad::GradientClip, l::Layer)
    gradient_clip!(l.ΔW, grad.threshold)
end

function backprop!(grad::GradientClip, l::RecurrentLayer)
    gradient_clip!(l.ΔW,  grad.threshold)
    gradient_clip!(l.ΔWh, grad.threshold)
end

function gradient_clip!(g::Weight, limit::Float32)
    nm = norm(g,1)
    if nm > limit
        rate = limit / nm
        ng = @in1! g .* rate
    end
    g
end
