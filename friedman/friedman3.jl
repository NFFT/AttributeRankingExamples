using ANOVAapprox
using LinearAlgebra
using Random
using JLD2 
using FileIO
using Distributions

include("friedman.jl")

M_train = 200
M_test = 1000

@load "X_test_3.jld2"
@load "X_train_3.jld2"
@load "y_test_3.jld2"
@load "y_train_3.jld2"
@load "noise_test_3.jld2"
@load "noise_train_3.jld2"

y_test += noise_test
y_train += noise_train

lambda = collect( 0.0:1.0:5.0 )
max_iter = 1000

###################################################

println( "==== Approximation Parameters ====" )
println( "M Train = ", M_train )
println( "M Test = ", M_test )
println( "lambda = ", lambda )
println( "max_iter = ", max_iter )

bws = Dict() 
element = 1

for N1 = 10:2:24
    for N2 = 2:2:2
        bws[element] = [N1,N2,2]
        global element += 1
    end
end

res = ANOVAapprox.testBandwidths( X_train, complex(y_train), X_test, complex(y_test), 3, bws; lambda=lambda )
min_res = findmin( res )

println("Table 1 (Attribute Rankings)")
println("==========")

for ( k, v ) in res 
    println( "bw: ", k[1], " lambda: ", k[2], " mse: ", v )
end

fun_approx = ANOVAapprox.nperiodic_approx( X_train, complex(y_train), 3, min_res[2][1] )
ANOVAapprox.approximate( fun_approx, max_iter=max_iter, lambda=[min_res[2][2],] )
println( "Attribute Ranking for ", min_res[2][1], ": ", ANOVAapprox.get_AttributeRanking( fun_approx, min_res[2][2] ) )

d = 3
X_train = X_train[1:d,:]
X_test = X_test[1:d,:]

###################################################

bws = Dict() 
element = 1

for N1 = 10:2:24
    for N2 = 2:2:4
        bws[element] = [N1,N2]
        global element += 1
    end
end


res = ANOVAapprox.testBandwidths( X_train, complex(y_train), X_test, complex(y_test), 2, bws; lambda=lambda )
min_res = findmin( res )

println("Table 2 (Approx 2d)")
println("==========")

fun_approx = ANOVAapprox.nperiodic_approx( X_train, complex(y_train), 2, min_res[2][1] )

for ( k, v ) in res 
    println( "bw: ", k[1], ", |I| = ", setSize( fun_approx.U, k[1] )," lambda: ", k[2], " mse: ", v )
end

ANOVAapprox.approximate( fun_approx, max_iter=max_iter, lambda=[min_res[2][2],] )
gsis = ANOVAapprox.get_GSI( fun_approx, min_res[2][2] )

println( "GSI for ", min_res[2][1], ":" )
for i = 1:length(fun_approx.U)
    println( fun_approx.U[i], ": ", gsis[i] )
end

###################################################

bws = Dict() 
element = 1

for N1 = 10:2:24
    for N2 = 2:2:4
        bws[element] = [N1,N2,2]
        global element += 1
    end
end

res = ANOVAapprox.testBandwidths( X_train, complex(y_train), X_test, complex(y_test), 3, bws; lambda=lambda )
min_res = findmin( res )

println("Table 2 (Approx 3d)")
println("==========")

fun_approx = ANOVAapprox.nperiodic_approx( X_train, complex(y_train), 3, min_res[2][1] )

for ( k, v ) in res 
    println( "bw: ", k[1], ", |I| = ", setSize( fun_approx.U, k[1] )," lambda: ", k[2], " mse: ", v )
end

ANOVAapprox.approximate( fun_approx, max_iter=max_iter, lambda=[min_res[2][2],] )
gsis = ANOVAapprox.get_GSI( fun_approx, min_res[2][2] )

println( "GSI for ", min_res[2][1], ":" )
for i = 1:length(fun_approx.U)
    println( fun_approx.U[i], ": ", gsis[i] )
end

function crossValidate( )
    fold = 100
    mses = zeros( fold )

    for i = 1:fold 
        train = generateData( 3, 200 )
        test = generateData( 3, 1000 )
        X_train = train[1]
        y_train = train[2] + train[3]
        X_test = test[1]
        y_test = test[2] + test[3]
        fun_approx = ANOVAapprox.nperiodic_approx( X_train, complex(y_train), 2, [12, 2], active_set=f3_active_set )
        ANOVAapprox.approximate( fun_approx, max_iter=1000, lambda=[1.0,0.0] )
        mses[i] = findmin(ANOVAapprox.get_MSE( fun_approx, X_test, complex(y_test)))[1]
    end

    return mses
end

