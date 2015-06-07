reload("src/NNN.jl")

const batch_size  = 20
const input_size  = 2
const output_size = 1
const hidden_size = 40

function toy_data(n)
    x = zeros(Float32, input_size,  n*batch_size)
    x[1,:] = rand(n*batch_size)
    y = zeros(Float32, output_size, n*batch_size)

    for i in 0:batch_size:((n-1)*batch_size)
        a = floor(Int64, rand() * batch_size)+1
        b = floor(Int64, rand() * batch_size)+1
        total = 0.0f0
        for j in 1:batch_size
            pos = i + j
            if j == a || j == b
                x[2, pos] = 1.0f0
                total += x[1, pos]
            end
            y[1,pos] = total
        end
    end

    (x, y)
end

function rnn_train(xtrn, ytrn, xtst, ytst, iter)
    net = NNN.Network(batch_size)
    NNN.push!(net,
             NNN.SimpleRecurrentLayer(batch_size, input_size, hidden_size;
                                     activation=NNN.LeakyRelu()),
             NNN.AdaGrad(input_size, hidden_size;
                        ε=0.1,
                        μ=0.05,
                        λ=0.0000001),
             NNN.AdaGrad(hidden_size, hidden_size;
                        ε=0.1,
                        μ=0.05,
                        λ=0.000001))
    NNN.push!(net,
             NNN.OutputLayer(batch_size, hidden_size, output_size;
                             activation=NNN.IdentityAct(),
                             loss=NNN.rnn_quadratic_cost),
                             post_filters=[NNN.GradientClip(10.0f0)]
             NNN.SGD(;λ=0.000001,
                    ε=0.05))

    @time NNN.train(xtrn, ytrn, net; iter=iter, batch_size=batch_size)
    errors = NNN.test(xtst,  ytst, net; batch_size=batch_size)

    (net, errors)
end

function profile_rnn(xtrn, ytrn, xtst, ytst, iter)
    Profile.clear()
    Profile.init()

    net, errors = @profile rnn_train(xtrn, ytrn, xtst, ytst, iter)

    open("/tmp/prof.dat", "w") do io
        Profile.print(io)
    end

    (net,errors)
end

xtrn, ytrn = toy_data(5000);
xtst, ytst = toy_data(1000);

net,errors = profile_rnn(xtrn, ytrn, xtst, ytst, 1); errors

# NNN.train(xtrn, ytrn, net; iter=10, batch_size=batch_size)
