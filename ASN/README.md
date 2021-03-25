# Airfoil Self-Noise Dataset

This folder contains the code necessary to replicate our experiments with the Airfoil Self-Noise data. The file `asn_data.jld2` contains the data from [UCI Machine Learning Repository: Airfoil Self-Noise Data Set](https://archive.ics.uci.edu/ml/datasets/Airfoil+Self-Noise) processed as a Julia Matrix. In the following we explain the Julia code files. 

## asn_active_set_detection.jl

This file will produce the `asn_activeSet.jld2` file containing the active set of ANOVA terms by searching for a model with low RMSE and removing terms by a cutoff parameter epsilon. This is performed on a single split of the data contained in `asn_example.jld2`.

## asn_bw_detection.jl

This file is resposible for detection the optimal bandwidth parameters by cross validation. 

## asn_validation.jl

This file validates the model on 100 random splits of training and test data. The console output of executing this file was saved in `asn_validation.txt`.

## asn_attribute_ranking.jl

This file produces the attribute ranking of our final approximation model.

## Packages

The code requires different Julia packages which can be installed using the Julia Package manager. The package LibTest was developed specifically for the paper and is not available through the package manager. We provide the code file `LibTest.jl` in the folder LibTest. This folder can either be added to the Julia PATH or the library can be included by using `include("LOCALPATH/LibTest.jl")` and then `using .LibTest`. 

```julia
## in order to add it to path you have to do the following
using Distributed 
@everywhere push!(LOAD_PATH, "/YourPath/LibTest")
using LibTest

## in order to just add the module
include("/YourPath/LibTest.jl")
using .LibTest
```