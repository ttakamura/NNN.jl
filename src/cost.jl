# --------------------------------------------------
# Cost Functions
# --------------------------------------------------
function cross_entropy_cost(l::Layer, D::Signal, Y::Signal)  # D = 教師
    cost = 0.0
    for n in 1:size(D,2)
        for k in 1:size(Y,1)
            cost += D[k,n] * log(Y[k,n])
        end
    end
    -(cost / size(D,2))
end

function cross_entropy_cost(diff::Symbol, l::Layer, D::Signal, Y::Signal)
    copy!(l.ΔE, (Y - D))
end

# --------------------------------------------------
function quadratic_cost(l::Layer, D::Signal, Y::Signal)      # D = 教師
    norm(Y - D)^2 / 2
end

function quadratic_cost(diff::Symbol, l::Layer, D::Signal, Y::Signal)
    copy!(l.ΔE, (Y - D))
end

# --------------------------------------------------
function rnn_quadratic_cost(l::Layer, D::Signal, Y::Signal)      # D = 教師
    norm(Y[:,end] - D[:,end])^2 / 2
end

function rnn_quadratic_cost(diff::Symbol, l::Layer, D::Signal, Y::Signal)
    copy!(l.ΔE, zeros(Float32, size(Y)))
    l.ΔE[:,end] = Y[:,end] - D[:,end]
    l.ΔE
end

# --------------------------------------------------
function rnn_cross_entropy_cost(l::Layer, D::Signal, Y::Signal)  # D = 教師
    cost = 0.0
    for k in 1:size(Y,1)
        cost += D[k,end] * log(Y[k,end])
    end
    -(cost)
end

function rnn_cross_entropy_cost(diff::Symbol, l::Layer, D::Signal, Y::Signal)
    copy!(l.ΔE, zeros(Float32, size(Y)))
    l.ΔE[:,end] = Y[:,end] - D[:,end]
    l.ΔE
end
