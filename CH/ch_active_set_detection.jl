using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using JLD2
using FileIO
using Random
using Statistics
using LibTest

@load "ch_example.jld2"

lambda = collect(80.0:20.0:200.0)

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

v = LibTest.testBandwidths2d( ex, 12, lambda; N2_start=8 )

f = ANOVAapprox.nperiodic_approx( ex.X_train, ex.y_train, 2, v[1] )
ANOVAapprox.approximate( f, max_iter=200, lambda=[v[2],] )
println( "min RMSE: ", sqrt(ANOVAapprox.get_MSE(f, ex.X_test, ex.y_test, v[2])) )
println( "cor. BW: ", v[1] )

gsis = ANOVAapprox.get_GSI( f, v[2] )

println( "GSIs:" )

for i = 1:length(f.U)
    println( f.U[i], ": ", gsis[i] )
end

println("==========")
println("==========")
println("==========")

rmse = 100
AS = 0

for eps in 0.005:0.005:0.03

    activeSet = copy(ANOVAapprox.get_ActiveSet( f, [eps, eps] )[v[2]])
    f_eps = ANOVAapprox.nperiodic_approx( ex.X_train, ex.y_train, 2, v[1], active_set=activeSet )
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

@save "ch_activeSet.jld2" AS