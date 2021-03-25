using ANOVAapprox
using LinearAlgebra
using Random
using Distributions
using FileIO 
using JLD2

# scalings
s_1(x) = 100*x
s_2(x) = 520*pi*x+40*pi
s_4(x) = 10*x+1

# Friedman 1
function friedman1( x::Vector{Float64} )::Float64
    if ( minimum(x) < 0 ) || ( maximum(x) > 1 )
        error( "The nodes need to be between zero and one." )
    end

    return 10*sin(pi*x[1]*x[2])+20*((x[3]-0.5)^2)+10*x[4]+5*x[5]
end

# Friedman 2
function friedman2( x::Vector{Float64} )::Float64
    if ( minimum(x) < 0 ) || ( maximum(x) > 1 )
        error( "The nodes need to be between zero and one." )
    end

    return sqrt( (s_1(x[1]))^2 + (s_2(x[2])*x[3] - 1/(s_2(x[2])*s_4(x[4])) )^2 )
end

function friedman3( x::Vector{Float64} )::Float64
    if ( minimum(x) < 0 ) || ( maximum(x) > 1 )
        error( "The nodes need to be between zero and one." )
    end

    return atan( (s_2(x[2])*x[3] - 1/(s_2(x[2])*s_4(x[4]))) / s_1(x[1]) )
end

f1_active_set = Vector{Vector{Int64}}(undef, 7)
f1_active_set[1] = []
f1_active_set[2] = [1,]
f1_active_set[3] = [2,]
f1_active_set[4] = [3,]
f1_active_set[5] = [4,]
f1_active_set[6] = [5,]
f1_active_set[7] = [1,2]

f2_active_set = Vector{Vector{Int64}}(undef, 4)
f2_active_set[1] = []
f2_active_set[2] = [2,]
f2_active_set[3] = [3,]
f2_active_set[4] = [2,3]

f3_active_set = Vector{Vector{Int64}}(undef, 7)
f3_active_set[1] = []
f3_active_set[2] = [1,]
f3_active_set[3] = [2,]
f3_active_set[4] = [3,]
f3_active_set[5] = [1,2]
f3_active_set[6] = [1,3]
f3_active_set[7] = [2,3]

function setSize( U, N )
    freq = 0

    for u in U 
        if u != [] 
            freq += (N[length(u)]-1)^(length(u))
        end
    end

    return freq+1
end

function generateData( friedman_number, number_samples; rs=false )

    Random.seed!()

    if rs != false 
        Random.seed!(rs)
    end

    f = [ friedman1, friedman2, friedman3 ]
    d = [ 10, 4, 4 ]
    noise_variance = [ 1.0, 125.0, 0.1 ]

    dist = Normal( 0.0, noise_variance[friedman_number] )
    noise = rand( dist, number_samples )

    X = rand( d[friedman_number], number_samples )
    y = [ f[friedman_number](X[:,i]) for i = 1:number_samples ]

    return (X, y, noise)

end