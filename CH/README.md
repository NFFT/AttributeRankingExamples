# California Housing Dataset

This folder contains the code necessary to replicate our experiments with the California Housing data. The file `ch_data.jld2` contains the data from [Torgo: California Housing](https://www.dcc.fc.up.pt/~ltorgo/Regression/cal_housing.html) processed as a Julia Matrix. In the following we explain the Julia code files. 

## ch_active_set_detection.jl

This file will produce the `ch_activeSet.jld2` file containing the active set of ANOVA terms by searching for a model with low RMSE and removing terms by a cutoff parameter epsilon. This is performed on a single split of the data contained in `ch_example.jld2`.

## ch_bw_detection.jl

This file is resposible for detection the optimal bandwidth parameters by cross validation. 

## ch_validation.jl

This file validates the model on 100 random splits of training and test data. The console output of executing this file was saved in `ch_validation.txt`.

## ch_attribute_ranking.jl

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