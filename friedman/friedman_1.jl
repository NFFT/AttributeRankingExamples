using Distributed
using Printf
using JLD2
using Statistics
using FileIO
using Distributions
addprocs( 10 )

@everywhere using ANOVAapprox
@everywhere using GroupedTransforms
using Random

include("friedman.jl")

d = 10
ds = 2
M = 200
λs = [0.0,1.0,2.0,3.0,4.0]

@load "nodes_10d.jld2"
@load "noise_f1.jld2"
@load "nodes_test_10d.jld2"
@load "noise_f1_test.jld2"

data_values = Vector{Vector{Float64}}(undef, 10)

for i = 1:10
    X = data_nodes[i]
    data_values[i] = [friedman1( X[:, i] ) for i = 1:M] + data_noise[i]
end

data_values_test = Vector{Vector{Float64}}(undef, 10)

for i = 1:10
    X = data_nodes_test[i]
    data_values_test[i] = [friedman1( X[:, i] ) for i = 1:1000] + data_noise_test[i]
end

bw = [4,2]

println( "Friedman 1 Approximation with U^{(10,2)} and N = [4,2]" )

err_mse = 0.0
numFreq = 0
lambda = 0
ar = zeros( d )

for i = 1:10
    ads = ANOVAapprox.approx(data_nodes[i], data_values[i], ds, bw, "cos")
    global numFreq = get_NumFreq( ads.trafo.setting )
    ANOVAapprox.approximate(ads, lambda = λs)
    mses = ANOVAapprox.get_mse( ads, data_nodes_test[i], data_values_test[i] )

    min_mse = findmin(mses)
    global lambda = min_mse[2]

    global err_mse += 0.1 * min_mse[1]
    global ar += 0.1 .* ANOVAapprox.get_AttributeRanking( ads, lambda )
end

println( "|I| = ", numFreq )
println( "average mse = ", err_mse )
println( "lambda = ", lambda )
println( "attribute ranking = ", ar )

bw = [6,4]

println( "Friedman 1 Approximation with U^{(ar,0.001)} and N = [6,4]" )

err_mse = 0.0
numFreq = 0
lambda = 0
gsis = zeros( length( get_superposition_set(5,2) ) - 1 )

for i = 1:10
    ads = ANOVAapprox.approx(data_nodes[i][1:5,:], data_values[i], ds, bw, "cos")
    global numFreq = get_NumFreq( ads.trafo.setting )
    ANOVAapprox.approximate(ads, lambda = λs)
    mses = ANOVAapprox.get_mse( ads, data_nodes_test[i], data_values_test[i] )

    min_mse = findmin(mses)
    global lambda = min_mse[2]

    global err_mse += 0.1 * min_mse[1]
    global gsis += 0.1 .* ANOVAapprox.get_GSI( ads, lambda )
end

println( "|I| = ", numFreq )
println( "average mse = ", err_mse )
println( "lambda = ", lambda )
println( "gsis = " )

for i = 1:length(gsis)
    print( "(",i,",",gsis[i],") ")
end

println( "..." )

bws = [ [4,4], [6,4], [8,4], [10,4], [4,6], [6,6], [8,6], [10,6], [4,8], [6,8], [8,8], [10,8] ]

for bw in bws 
    err_mse_bw = 0.0
    numFreq_bw = 0
    lambda_bw = 0

    for i = 1:10
        ads = ANOVAapprox.approx(data_nodes[i][1:5,:], data_values[i], f1_active_set, bw, "cos")
        numFreq_bw = get_NumFreq( ads.trafo.setting )
        ANOVAapprox.approximate(ads, lambda = λs)
        mses = ANOVAapprox.get_mse( ads, data_nodes_test[i], data_values_test[i] )
        min_mse = findmin(mses)
        lambda_bw = min_mse[2]

        err_mse_bw += 0.1 * min_mse[1]
    end

    s = string( "\$", bw, "\$ & \$", numFreq_bw, "\$ & \$",lambda_bw,"\$ & \$", err_mse_bw, "\$ \\\\ " )

    println( s )
end

bw = [6,4]
mse_vec = zeros( 100 )

for i = 1:100 
    rng = MersenneTwister( rand(1000:9999) )
    dist = Normal( 0.0, 1.0 )

    X = rand(rng, 10, 200) 
    X_test = rand(rng, 10, 1000)
    y = [friedman1( X[:, i] ) for i = 1:200] + rand( rng, dist, 200 )
    y_test = [friedman1( X_test[:, i] ) for i = 1:1000] + rand( rng, dist, 1000 )

    ads = ANOVAapprox.approx(X[1:5,:], y, f1_active_set, bw, "cos")
    ANOVAapprox.approximate(ads, lambda = [0.0,])
    mse_vec[i] = ANOVAapprox.get_mse( ads, X_test[1:5,:], y_test, 0.0 )
end

println( "Median over 100 experiments with N = ", bw, ": ", median(mse_vec) )