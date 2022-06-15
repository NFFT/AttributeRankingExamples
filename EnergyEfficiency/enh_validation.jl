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

@load "energy_data.jld2"
@load "enh_activeSet.jld2"

X = dM[:,1:8]
y = dM[:,9]

dtx = fit(UnitRangeTransform, X, dims=1)

X_scaled = StatsBase.transform(dtx, X)
y_scaled = y

bw = [ 100, 6 ]
lambda = 20.0

ex = LibTest.example( X_scaled, complex(y_scaled), 0.3 )

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

rmses = zeros(cv_n_times)

for i = 1:cv_n_times
    println( "==== CV Set ", i ," ====" )
    cex = LibTest.example( X_scaled, complex(y_scaled), 0.3 )
    f = ANOVAapprox.approx( cex.X_train, cex.y_train, AS, bw, "cos")
    ANOVAapprox.approximate( f, max_iter=200, lambda=[lambda,] )
    rmses[i] = sqrt(ANOVAapprox.get_mse(f, cex.X_test, cex.y_test, lambda))
    println( "RMSE: ", rmses[i] )
end

println( "======" )
println( "======" )
println( "======" )

println( "Median: ", median(rmses) )

