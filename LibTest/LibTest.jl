module LibTest

using CSV
using StatsBase
using DataFrames
using ANOVAapprox
using LinearAlgebra
using Random

function getN1fromN2( N2 )
    return (N2 - 1)*(N2 - 1) + 1
end

function train_test_split( M, V, p )
    percentage_test = p
    perm = randperm(size(M,1))
    n_train = Integer(ceil((1-percentage_test)*length(perm)))

    perm_train = perm[1:n_train]
    perm_test = perm[n_train+1:end]

    X_train = M[ perm_train, : ]
    X_test = M[ perm_test, : ]
    y_train = V[ perm_train ]
    y_test = V[ perm_test ]

    X_train = convert(Matrix, transpose(X_train))
    X_test = convert(Matrix, transpose(X_test))

    return ( X_train, y_train, X_test, y_test )
end
  
mutable struct example 
    X::Matrix{Float64}
    y::Vector{ComplexF64}
    X_train::Matrix{Float64}
    y_train::Vector{ComplexF64}
    X_test::Matrix{Float64}
    y_test::Vector{ComplexF64}
    p_test::Float64
  
    function example( X::Matrix{Float64}, y::Vector{ComplexF64}, p_test::Float64 ) 
        data = train_test_split( X, y, p_test )
        return new(X, y, data[1], data[2], data[3], data[4], p_test )
    end 
end

function testBandwidths2d( ex::example, N2_stop::Integer, lambdas::Vector{Float64}; active_set=false, N2_start::Integer=2 )
    bws = Dict()
    i = 1 

    for N2 = N2_start:2:N2_stop 
        bws[i] = [getN1fromN2( N2 ), N2]
        i += 1
    end

    res = ANOVAapprox.testBandwidths( ex.X_train, ex.y_train, ex.X_test, ex.y_test, 2, bws; lambda=lambdas, max_iter=200, active_set=active_set )

    for ( k, v ) in res 
        println( "bw: ", k[1], " lambda: ", k[2], " rmse: ", sqrt(v) )
    end

    min_res = findmin( res )

    return min_res[2]
end

function testBandwidths1d( ex::example, N1::Vector{Int64}, lambdas::Vector{Float64}; active_set=false )
    bws = Dict()
    i = 1 

    for j = 1:length(N1)
        bws[i] = [N1[j],]
        i += 1
    end

    res = ANOVAapprox.testBandwidths( ex.X_train, ex.y_train, ex.X_test, ex.y_test, 1, bws; lambda=lambdas, max_iter=200, active_set=active_set )

    for ( k, v ) in res 
        println( "bw: ", k[1], " lambda: ", k[2], " rmse: ", sqrt(v) )
    end

    min_res = findmin( res )

    return min_res[2]
end

end