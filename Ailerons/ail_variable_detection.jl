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

@load "ail_data.jld2"

X = dM[:,1:40]
y = dM[:,41]

dtx = fit(UnitRangeTransform, X, dims=1)
dty = fit(UnitRangeTransform, y)
X_scaled = StatsBase.transform(dtx, X)
y_scaled = StatsBase.transform(dty, y)

lambda = [1.0, 5.0, 10.0, 15.0]

ex = LibTest.example( X_scaled, complex(y_scaled), 0.5 )

println( "==== Approximation Parameters ====" )
println( "|X_train| = ", size(ex.X_train, 2) )
println( "|X_test| = ", size(ex.X_test, 2) )
println( "lambda = ", lambda )

ar_mean = zeros( 40 )

for i = 1:cv_n_times
    println( "==== CV Set ", i ," ====" )
    cex = LibTest.example( X_scaled, complex(y_scaled), 0.5 )
    f = ANOVAapprox.nperiodic_approx( cex.X_train, cex.y_train, 1, [20,] )
    ANOVAapprox.approximate( f, max_iter=200, lambda=[10.0,] )
    r = ANOVAapprox.get_AttributeRanking( f, 10.0 ) ./ cv_n_times
    global ar_mean += r
end

println( "======" )
println( "======" )
println( "======" )

vars = sortperm(ar_mean, rev=true)
vars_rmses = zeros(15)

for v = 10:24 
    act_vars = vars[1:v]
    for i = 1:20 
        println( "==== CV Set ", i ," ====" )
        cex = LibTest.example( X_scaled[:,act_vars], complex(y_scaled), 0.5 )
        f = ANOVAapprox.nperiodic_approx( cex.X_train, cex.y_train, 1, [20,] )
        ANOVAapprox.approximate( f, max_iter=200, lambda=[10.0,] )
        vars_rmses[v-9] += sqrt(ANOVAapprox.get_MSE(f, cex.X_test, cex.y_test, 10.0))/20
    end
end

v = vars[1:11]
@save "ail_vars.jld2" v