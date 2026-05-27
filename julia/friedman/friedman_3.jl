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

d = 4
ds = 4
M = 200
λs = [0.0,1.0,2.0,3.0,4.0]

@load "nodes_4d.jld2"
@load "noise_f3.jld2"
@load "nodes_test_4d.jld2"
@load "noise_f3_test.jld2"

data_values = Vector{Vector{Float64}}(undef, 10)

for i = 1:10
    X = data_nodes[i]
    data_values[i] = [friedman3( X[:, j] ) for j = 1:M] + data_noise[i]
end

data_values_test = Vector{Vector{Float64}}(undef, 10)

for i = 1:10
    X = data_nodes_test[i]
    data_values_test[i] = [friedman3( X[:, j] ) for j = 1:1000] + data_noise_test[i]
end

bw = [4,2,2,2]

println( "Friedman 3 Approximation with U^{(4,4)} and N = [4,2,2,2]" )

err_mse = 0.0
numFreq = 0
lambda = 0
ar = zeros( 4 )
U = 0

for i = 1:10
    ads = ANOVAapprox.approx(data_nodes[i], data_values[i], ds, bw, "cos")
    global U = ads.U[2:end]
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
println( "ar = " )

for i = 1:4
    print( "(",i,",",ar[i],") ")
end

println( "..." )

bws = Vector{Vector{Int}}(undef, 12)
idx = 1

for N1 in [8,10,12,14]
    for N2 in [2,4,6]
        bws[idx] = [N1,N2]
        global idx += 1
    end
end

for bw in bws 
    err_mse_bw = 0.0
    numFreq_bw = 0
    lambda_bw = 0

    for i = 1:10
        ads = ANOVAapprox.approx(data_nodes[i], data_values[i], f3_active_set, bw, "cos")
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

bw = [12,2]
mse_vec = zeros( 100 )

for i = 1:100 
    rng = MersenneTwister( rand(1000:9999) )
    dist = Normal( 0.0, 0.1 )

    X = rand(rng, 4, 200) 
    X_test = rand(rng, 4, 1000)
    y = [friedman3( X[:, i] ) for i = 1:200] + rand( rng, dist, 200 )
    y_test = [friedman3( X_test[:, i] ) for i = 1:1000] + rand( rng, dist, 1000 )

    ads = ANOVAapprox.approx(X, y, f3_active_set, bw, "cos")
    ANOVAapprox.approximate(ads, lambda = [4.0,])
    mse_vec[i] = ANOVAapprox.get_mse( ads, X_test, y_test, 4.0 )
end

println( "Median over 100 experiments with N = ", bw, ": ", median(mse_vec) )
