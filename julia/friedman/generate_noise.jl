using Random
using JLD2
using Distributions

noise_variance = [ 1.0, 125.0, 0.1 ]

for j = 1:3 
    data_noise = Vector{Vector{Float64}}(undef, 10)
    for i = 1:10
        rng = MersenneTwister( rand(1000:9999) )
        dist = Normal( 0.0, noise_variance[j] )
        data_noise[i] = rand( rng, dist, 200 )
    end
    jldsave( string("noise_f",j,".jld2"); data_noise )
end

for j = 1:3 
    data_noise_test = Vector{Vector{Float64}}(undef, 10)
    for i = 1:10
        rng = MersenneTwister( rand(1000:9999) )
        dist = Normal( 0.0, noise_variance[j] )
        data_noise_test[i] = rand( rng, dist, 1000 )
    end
    jldsave( string("noise_f",j,"_test.jld2"); data_noise_test )
end
