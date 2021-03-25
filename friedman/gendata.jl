using ANOVAapprox
using LinearAlgebra
using Random
using JLD2 
using FileIO
using Distributions

include("friedman.jl")

M_train = 200
M_test = 1000

for i = 1:3
    train = generateData( i, M_train )
    test = generateData( i, M_test )
    X_train = train[1]
    y_train = train[2]
    noise_train = train[3]
    X_test = test[1]
    y_test = test[2]
    noise_test = test[3]
    @save string( "X_train_", i, ".jld2" ) X_train
    @save string( "y_train_", i, ".jld2" ) y_train
    @save string( "noise_train_", i, ".jld2" ) noise_train
    @save string( "X_test_", i, ".jld2" ) X_test
    @save string( "y_test_", i, ".jld2" ) y_test
    @save string( "noise_test_", i, ".jld2" ) noise_test
end