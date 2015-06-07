using Devectorize

function update!(net::Network)
    for i in 1:length(net.layers)
        update!(net.layers[i], net.optimizers[i], net.rnn_optimizers[i])
    end
end

function update!(layer::Layer, opt::Optimizer, dummy::Optimizer)
    update_weight!(opt, layer.W, layer.ΔW)
    update_bias!(opt, layer.b, layer.Δb)
end

function update!(layer::RecurrentLayer, opt::Optimizer, rnn_opt::Optimizer)
    update_weight!(opt, layer.W, layer.ΔW)
    update_weight!(rnn_opt, layer.Wh, layer.ΔWh)
    update_bias!(opt, layer.b, layer.Δb)
end

function update_bias!(sgd::Optimizer, b, Δb)
    db = @in1! Δb .* -0.3
    nb = @in1! b   .+ Δb
end

# --------------------------------------------------------------------------------------------
immutable SGD <: Optimizer
    λ::Float32            # weight decrese
    ε::Float32            # learning rate

    function SGD(;λ=0.01, ε=1.0)
        new(λ, ε)
    end
end

function update_weight!(sgd::SGD, W, ΔW)
    ε=sgd.ε; λ=sgd.λ

    @inbounds for col=1:size(W,2), row=1:size(W,1)
        ΔW[row, col] += W[row, col] * λ
        ΔW[row, col] *= -ε
        W[row, col]   += ΔW[row, col]
    end
end

# --------------------------------------------------------------------------------------------
immutable AdaGrad <: Optimizer
    λ::Float32  # weight decrese
    ε::Float32  # learning rate
    μ::Float32  # RMSprop
    G::Weight    # sum of gradients
    H::Weight    # sum of gradients^2

    function AdaGrad(in_size, out_size; ε=0.0, λ=0.01, μ=0.0)
        new(λ, ε, μ, zeros(out_size, in_size), zeros(out_size, in_size))
    end
end

function update_weight!(ada::AdaGrad, W, ΔW)
    ε=ada.ε; λ=ada.λ; μ=ada.μ

    new_g = @in1! ada.G .+ ΔW

    if μ > 0.0
        copy!(ada.H, (μ * (ΔW .^ 2) + (1.0 - μ) * ada.H)) # RMSprop
    else
        new_h = @in1! ada.H .+ (ΔW .^ 2)                    # AdaGradRDA
    end

    H = sqrt(ada.H)
    for col=1:size(W,2), row=1:size(W,1)
        W[row,col] += (-ε * ΔW[row,col]) / (1.0 + H[row,col])

        # FOBOS
        if λ > 0.0
            if abs(W[row,col]) < λ
                W[row,col] = 0.0
            elseif W[row,col] > 0.0
                W[row,col] -= λ
            else
                W[row,col] += λ
            end
        end
    end
end


# --------------------------------------------------------------------------------------------
type AdaGradRDA <: Optimizer
    λ::Float32  # weight decrese
    μ::Float32  # RMSprop
    G::Weight    # sum of gradients
    H::Weight    # sum of gradients^2
    num_of_grads::Int64 # number of gradients

    function AdaGradRDA(in_size, out_size; λ=0.01, μ=0.0)
        new(λ, μ, zeros(out_size, in_size), zeros(out_size, in_size), 0)
    end
end

function update_weight!(ada::AdaGradRDA, W, ΔW)
    λ=ada.λ; μ=ada.μ

    ada.num_of_grads += 1
    ada.G            += ΔW

    if μ > 0.0
        ada.H = μ * (ΔW .^ 2) + (1.0 - μ) * ada.H  # RMSprop
    else
        ada.H += ΔW .^ 2                             # AdaGradRDA
    end

    G = ada.G ./ ada.num_of_grads

    for col=1:size(W,2), row=1:size(W,1)
        if abs(G[row,col]) < λ
            W[row,col] = 0.0
        elseif G[row,col] > λ
            W[row,col] = (-1.0 * G[row,col] * ada.num_of_grads + λ) / (1.0 + ada.H[row,col])
        else
            W[row,col] = (-1.0 * G[row,col] * ada.num_of_grads - λ) / (1.0 + ada.H[row,col])
        end
    end
end
