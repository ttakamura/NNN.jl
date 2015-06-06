reload("/Users/tatsuya/src/github.com/ttakamura/julia-sandbox/deep_nn/deep_nn.jl")

const batch_size  = 20
const input_size  = 10
const output_size = 10
const hidden_size = 30

function dummy_data(n)
    x = rand(Float32, input_size,  n) * 1.3
    y = zeros(Float32, output_size, n)
    for i in 1:n
        k = i%10+1
        # k = floor(Int64, (rand() * 10)) + 1
        y[k,i] = 1.0f0
        x[k,i] = 1.0f0
    end
    (x, y)
end

function rnn_train(xtrn, ytrn, xtst, ytst)
    net = NNN.Network(batch_size)
    NNN.push!(net,
             NNN.SimpleRecurrentLayer(batch_size, input_size, hidden_size;
                                     activation=NNN.relu),
             NNN.AdaGrad(input_size, hidden_size;
                        #μ=0.2,
                        ε=0.2,
                        λ=0.00001),
             NNN.AdaGrad(hidden_size, hidden_size;
                        #μ=0.2,
                        ε=0.2,
                        λ=0.00001))
    NNN.push!(net,
             NNN.OutputLayer(batch_size, hidden_size, output_size;
                            activation=NNN.softmax,
                            loss=NNN.cross_entropy_cost),
             NNN.SGD(;λ=0.0000001,
                    ε=0.2))

    NNN.train(xtrn, ytrn, net; iter=20,  batch_size=batch_size)
    errors = NNN.test(xtst,  ytst, net; batch_size=batch_size)

    (net, errors)
end

xtrn, ytrn = dummy_data(1000);
xtst, ytst = dummy_data(1000);
net, errors = rnn_train(xtrn, ytrn, xtst, ytst); errors
