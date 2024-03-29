using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using JLD2
using FileIO
using Random
using Statistics
using LibTest

cv_n_times = 20

@load "energy_data.jld2"
@load "enh_activeSet.jld2"

X = dM[:,1:8]
y = dM[:,9]

dtx = fit(UnitRangeTransform, X, dims=1)

X_scaled = StatsBase.transform(dtx, X)
y_scaled = y

lambda = collect(0.0:20.0:200.0)

ex = LibTest.example( X_scaled, complex(y_scaled), 0.3 )

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

bws = Dict()
i = 1

for N1 = 10:20:150
    for N2 in 4:2:8
        bws[i] = [N1,N2]
        global i += 1
    end
end

rmses = Dict()

for (i,bw) in bws 
    rmses[bw] = 0.0
end

for i = 1:cv_n_times
    println( "==== CV Set ", i ," ====" )
    cex = LibTest.example( X_scaled, complex(y_scaled), 0.3 )
    res = LibTest.testBandwidths( cex.X_train, cex.y_train, cex.X_test, cex.y_test, 2, bws; lambda=lambda, max_iter=200, active_set=AS )
    for ( k, v ) in res
        rmses[k[1]] += sqrt(v)/cv_n_times
        println( "bw: ", k[1], " lambda: ", k[2], " rmse: ", sqrt(v) )
    end
end

println( "======" )
println( "======" )
println( "======" )

println( findmin(rmses) )

