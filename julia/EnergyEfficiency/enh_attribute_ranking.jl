using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using JLD2
using FileIO
using Random
using Statistics
using LibTest

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
f = ANOVAapprox.approx( ex.X_train, ex.y_train, AS, bw, "cos" )
ANOVAapprox.approximate( f, max_iter=200, lambda=[lambda,] )
println( "RMSE: ", sqrt(ANOVAapprox.get_mse(f, ex.X_test, ex.y_test, lambda)) )
println( "Attribute Ranking: " )

r = ANOVAapprox.get_AttributeRanking(f, lambda)

for i = 1:length(r) 
    print( "( ", i, ", ", r[i], " )" )
end
