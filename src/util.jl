function train(Xs::Signal, Ys::Signal, net::Network; iter=50, batch_size=10)
    errors = Float32[]
    for t in 1:iter
        batch_num = floor(Int64, (size(Xs, 2) / batch_size)) - 1
        Y = Ys[:, 1:2]
        for i in 1:batch_num
            ids = [1:batch_size;] + (i * batch_size)
            X   = Xs[:, ids]
            Y   = Ys[:, ids]

            backprop!(net, X, Y)
            update!(net)

            if (i % 500) == 0
                report(net, errors, Y, ids)
           end
        end
    end
    errors
end

function test(Xs::Signal, Ys::Signal, net::Network; iter=10, batch_size=10)
    miss = 0
    for t in 1:iter
        ids = [1:batch_size;] + (t * batch_size)
        X   = Xs[:, ids]
        Y   = Ys[:, ids]

        Z = forward!(net, X)

        for n in 1:batch_size
            ys, yn = findmax(Y[:, n])
            zs, zn = findmax(Z[:, n])
            if yn != zn
                miss += 1
            end
        end
    end
    1.0 - (miss ./ (iter * batch_size))
end

function report(net, errors, Y, ids)
    l = net.layers[end]
    cost = l.loss(l, Y, l.Z)
    Base.push!(errors, cost)
    println("$(ids[1]) ~ $(ids[end])", " ",
            printfp("cost", cost),     "\t\t|\t",
            report(net.layers[1]),     "\t|\t",
            report(net.layers[2])
    )
end

function report(l::Layer)
    join([printfp("E", l.ΔE),
          printfp("W", l.W),
          printfp("Z", l.Z),
          printfp("X", l.X)], "  ")
end

function report(l::SimpleRecurrentLayer)
    join([printfp("E",  l.ΔE),
          printfp("W",  l.W),
          printfp("Wh", l.Wh),
          printfp("Z",  l.Z),
          printfp("X",  l.X)], "  ")
end

function printfp(label::AbstractString, x::FloatingPoint)
    join([label, ":", @sprintf("%.2f", x)])
end

function printfp(label::AbstractString, x::Matrix{Float32})
    printfp(label, sum(x))
end
