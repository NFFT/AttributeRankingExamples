# Friedman Functions

This folder contains the code necessary to replicate our experiments with the Friedman functions. The files `X_train_i.jld2` and `y_train_i.jld2` contain the training data for the functions i=1,2,3. The files `X_test_i.jld2` and `y_test_i.jld2` contain the corresponding test data and the generated noise is in `noise_train_i.jld2` and `noise_test_i.jld2`. 

## friedman.jl

This functions contains code to evaluate the functions and generate data.

## gendata.jl

This file was used to generated the `.jld2` mentioned above for each function.

## friedman_i.jl

This code is used to produce the model and provides a function to validate it on randomly generated training and testsets for each of the three Friedman functions.

## Packages

The code requires different Julia packages which can be installed using the Julia Package manager.
