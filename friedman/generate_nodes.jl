using Random
using JLD2

data_nodes = Vector{Matrix{Float64}}(undef, 10)

for i = 1:10
    rng = MersenneTwister( rand(1000:9999) )
    data_nodes[i] = rand(rng, 10, 200) 
end

jldsave( "nodes_10d.jld2"; data_nodes )

data_nodes = Vector{Matrix{Float64}}(undef, 10)

for i = 1:10
    rng = MersenneTwister( rand(1000:9999) )
    data_nodes[i] = rand(rng, 4, 200) 
end

jldsave( "nodes_4d.jld2"; data_nodes )

data_nodes_test = Vector{Matrix{Float64}}(undef, 10)

for i = 1:10
    rng = MersenneTwister( rand(1000:9999) )
    data_nodes_test[i] = rand(rng, 10, 1000) 
end

jldsave( "nodes_test_10d.jld2"; data_nodes_test )

data_nodes_test = Vector{Matrix{Float64}}(undef, 10)

for i = 1:10
    rng = MersenneTwister( rand(1000:9999) )
    data_nodes_test[i] = rand(rng, 4, 1000) 
end

jldsave( "nodes_test_4d.jld2"; data_nodes_test )


