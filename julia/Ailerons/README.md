# Ailerons Dataset

This folder contains the code necessary to replicate our experiments with the Ailerons data. The file `ail_data.jld2` contains the data from [Torgo: Ailerons](https://www.dcc.fc.up.pt/~ltorgo/Regression/ailerons.html) processed as a Julia Matrix. In the following we explain the Julia code files. 

## ail_variable_detection.jl

This file will produce the `ail_vars.jld2` file containing the active set of variables determined by cross validation over the number of variables.

## ail_active_set_detection.jl

This file will produce the `ail_activeSet.jld2` file containing the active set of ANOVA terms by searching for a model with low RMSE and removing terms by a cutoff parameter epsilon. This is performed on a single split of the data contained in `ail_example.jld2`.

## ail_bw_detection.jl

This file is resposible for detection the optimal bandwidth parameters by cross validation. 

## ail_validation.jl

This file validates the model on 100 random splits of training and test data. The console output of executing this file was saved in `ail_validation.txt`.

## ail_attribute_ranking.jl

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