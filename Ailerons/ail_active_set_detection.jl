using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using JLD2
using FileIO
using Random
using Statistics
using LibTest

@load "ail_example.jld2"
@load "ail_vars.jld2"

ex.X_train = ex.X_train[v,:]
ex.X_test = ex.X_test[v,:]
ex.X = ex.X[:,v]

lambda = collect(5.0:5.0:20.0)

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

f = ANOVAapprox.nperiodic_approx( ex.X_train, ex.y_train, 2, [12,2] )
ANOVAapprox.approximate( f, max_iter=200, lambda=lambda )
res = findmin(ANOVAapprox.get_MSE(f, ex.X_test, ex.y_test))

println( "min RMSE: ", sqrt(res[1]) )
println( "cor. BW: ", [12,2], " lambda: ", res[2] )

gsis = ANOVAapprox.get_GSI( f, res[2] )

println( "GSIs:" )

for i = 1:length(f.U)
    println( f.U[i], ": ", gsis[i] )
end

println("==========")
println("==========")
println("==========")

rmse = 100
AS = 0

for eps in 0.001:0.001:0.01

    activeSet = copy(ANOVAapprox.get_ActiveSet( f, [eps, eps] )[res[2]])
    f_eps = ANOVAapprox.nperiodic_approx( ex.X_train, ex.y_train, 2, [12,2], active_set=activeSet )
    ANOVAapprox.approximate( f_eps, max_iter=200, lambda=lambda )
    m = findmin( ANOVAapprox.get_MSE(f_eps, ex.X_test, ex.y_test) )

    if sqrt(m[1]) < rmse
        global AS = copy(activeSet)
        global rmse = sqrt(m[1])
    end

    println( "cutoff: ", eps)
    println( "lambda: ", m[2] )
    println( "RMSE: ", sqrt(m[1]) )
    println("==========")

end

@save "ail_activeSet.jld2" AS