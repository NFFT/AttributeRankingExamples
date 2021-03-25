using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using JLD2
using FileIO
using Random
using Statistics
using LibTest

cv_n_times = 100

@load "asn_data.jld2"
@load "asn_activeSet.jld2"

X = dM[:,1:5]
y = dM[:,6]

dtx = fit(UnitRangeTransform, X, dims=1)

X_scaled = StatsBase.transform(dtx, X)
y_scaled = y

bw = [ 200, 30 ]
lambda = 100.0

ex = LibTest.example( X_scaled, complex(y_scaled), 0.2 )

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

rmses = zeros(cv_n_times)

for i = 1:cv_n_times
    println( "==== CV Set ", i ," ====" )
    cex = LibTest.example( X_scaled, complex(y_scaled), 0.2 )
    f = ANOVAapprox.nperiodic_approx( cex.X_train, cex.y_train, 2, bw, active_set=AS )
    ANOVAapprox.approximate( f, max_iter=200, lambda=[lambda,] )
    rmses[i] = sqrt(ANOVAapprox.get_MSE(f, cex.X_test, cex.y_test, lambda))*sqrt(length(cex.y_test))/norm(cex.y_test)
    println( "rel. error: ", rmses[i] )
end

println( "======" )
println( "======" )
println( "======" )

println( "Median: ", median(rmses) )

