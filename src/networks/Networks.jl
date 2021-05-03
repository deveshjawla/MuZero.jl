function dict_to_cpu(Dict())
    return Dict()
end

function scalar_to_support(x, support_size)
    return []
end

function support_to_scalar(logits, support_size)
    return nothing
end

struct FullyConnected
action_space_size
full_support_size
representation_network
dynamics_hidden_state_network
dynamics_reward_network
prediction_policy_network
prediction_value_network
end

struct Network
action_space_size
full_support_size
block_output_size_reward
block_output_size_value
block_output_size_policy
representation_network
dynamics_network
prediction_network
end


module Networks

export Network, FeedForwardHP, ResNetHP, CyclicNesterov, Adam, OptimiserSpec, Network

using Parameters: @with_kw
using Statistics: mean
using CUDA
import Flux, Functors

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten
using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, MeanPool, MaxPool, AdaptiveMeanPool
import Zygote

# Flux.@functor does not work due to Network being parametric
function Flux.functor(nn::Net) where Net <: Network
  children = (nn.common, nn.first_head, nn.second_head)
  constructor = cs -> Net(nn.gspec, nn.hyper, cs...)
  return (children, constructor)
end

# This should be included in Flux
function lossgrads(f, args...)
  val, back = Zygote.pullback(f, args...)
  grad = back(Zygote.sensitivity(val))
  return val, grad
end


#####
##### Interface
#####

function convert_input_tuple(nn::Network, input::Tuple)
  return map(input) do arr
    array_on_gpu(nn.first_head[end].b) ? Flux.gpu(arr) : arr
  end
end

function convert_output_tuple(output::Tuple)
  return map(output) do arr
    Flux.cpu(arr)
  end
end

"""
    forward(::Network, states)

Compute the forward pass of a network on a batch of inputs.

Expect a `Float32` tensor `states` whose batch dimension is the last one.

Return a `(P, V)` triple where:

  - `P` is a matrix of size `(num_actions, batch_size)`. It is allowed
    to put weight on invalid actions (see [`evaluate`](@ref)).
  - `V` is a row vector of size `(1, batch_size)`
"""
function forward(nn::Network, input)
  if isdefined(nn,nn.downsample)
    input=nn.downsample(input)
  end
  state = nn.common(input)
  first = nn.first_head(state)
  if isdefined(nn, nn.second_head)
    second = nn.second_head(state)
    return (first, second)
  else
    return first
  end
end


# Reimplementation of what used to be Flux.prefor, does not visit leaves
function foreach_flux_node(f::Function, x, seen = IdDict())
  Functors.isleaf(x) && return
  haskey(seen, x) && return
  seen[x] = true
  f(x)
  for child in Flux.trainable(x)
    foreach_flux_node(f, child, seen)
  end
end
"""
    regularized_params(::Network)

Return the collection of regularized parameters of a network.
This usually excludes neuron's biases.
"""
function regularized_params(net::Network)
  ps = Flux.Params()
  foreach_flux_node(net) do p
    for r in regularized_params_(p)
      any(x -> x === r, ps) || push!(ps, r)
    end
  end
  return ps
end
regularized_params_(l) = []
regularized_params_(l::Flux.Dense) = [l.W]
regularized_params_(l::Flux.Conv) = [l.weight]

"""
    gc(::Network)

Perform full garbage collection and empty the GPU memory pool.
"""
gc(::Network)=GC.gc(true) # CUDA.reclaim()

# Optimizers and training

"""
    OptimiserSpec

Abstract type for an optimiser specification.
"""
abstract type OptimiserSpec end

"""
    CyclicNesterov(; lr_base, lr_high, lr_low, momentum_low, momentum_high)

SGD optimiser with a cyclic learning rate and cyclic Nesterov momentum.

  - During an epoch, the learning rate goes from `lr_low`
    to `lr_high` and then back to `lr_low`.
  - The momentum evolves in the opposite way, from high values
    to low values and then back to high values.
"""
@with_kw struct CyclicNesterov <: OptimiserSpec
  lr_base :: Float32
  lr_high :: Float32
  lr_low  :: Float32
  momentum_low :: Float32
  momentum_high :: Float32
end

"""
    Adam(;lr)

Adam optimiser.
"""
@with_kw struct Adam <: OptimiserSpec
  lr :: Float32
  #TODO??
  weight_decay::Float64 = 0.001
    momentum::Float64 = 0.9
    lr_init::Float64 = 0.05
    lr_decay_rate::Float64 = 0.1
    lr_decay_steps::Int64 = 350000
end

"""
    train!(callback, ::Network, opt::OptimiserSpec, loss, batches, n)

Update a given network to fit some data.
  - [`opt`](@ref OptimiserSpec) specifies which optimiser to use.
  - `loss` is a function that maps a batch of samples to a tracked real.
  - `data` is an iterator over minibatches.
  - `n` is the number of minibatches. If `length` is defined on `data`,
     we must have `length(data) == n`. However, not all finite
     iterators implement `length` and thus this argument is needed.
  - `callback(i, loss)` is called at each step with the batch number `i`
     and the loss on last batch.
"""
function train!(callback, nn::Network, opt::Adam, loss, data, n)
  optimiser = Flux.ADAM(opt.lr)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    callback(i, l)
  end
end

function train!(
    callback, nn::Network, opt::CyclicNesterov, loss, data, n)
  lr = CyclicSchedule(
    opt.lr_base,
    opt.lr_high,
    opt.lr_low, n=n)
  momentum = CyclicSchedule(
    opt.momentum_high,
    opt.momentum_low,
    opt.momentum_high, n=n)
  optimiser = Flux.Nesterov(opt.lr_low, opt.momentum_high)
  params = Flux.params(nn)
  for (i, d) in enumerate(data)
    l, grads = lossgrads(params) do
      loss(d...)
    end
    Flux.update!(optimiser, params, grads)
    optimiser.eta = lr[i]
    optimiser.rho = momentum[i]
    callback(i, l)
  end
end


#####
##### Derived functions
#####

"""
    num_parameters(::Network)

Return the total number of parameters of a network.
"""
function num_parameters(nn::Network)
  return sum(length(p) for p in Flux.params(nn))
end

"""
    num_regularized_parameters(::Network)

Return the total number of regularized parameters of a network.
"""
function num_regularized_parameters(nn::Network)
  return sum(length(p) for p in regularized_params(nn))
end

"""
    mean_weight(::Network)

Return the mean absolute value of the regularized parameters of a network.
"""
function mean_weight(nn::Network)
  sw = sum(sum(abs.(p)) for p in regularized_params(nn))
  sw = Flux.cpu(sw)
  return sw / num_regularized_parameters(nn)
end

"""
    forward_normalized(network::Network, states, actions_mask)

Evaluate a batch of vectorized states. This function is a wrapper
on [`forward`](@ref) that puts a zero weight on invalid actions.

# Arguments

  - `states` is a tensor whose last dimension has size `bach_size`
  - `actions_mask` is a binary matrix of size `(num_actions, batch_size)`

# Returned value

Return a `(P, V, Pinv)` triple where:

  - `P` is a matrix of size `(num_actions, batch_size)`.
  - `V` is a row vector of size `(1, batch_size)`.
  - `Pinv` is a row vector of size `(1, batch_size)`
     that indicates the total probability weight put by the network
     on invalid actions for each sample.

All tensors manipulated by this function have elements of type `Float32`.
"""
function forward_normalized(nn::Network, state, actions_mask)
  p, v = forward(nn, state)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1 .- sp
  return (p, v, p_invalid)
end

to_singletons(x) = reshape(x, size(x)..., 1)
from_singletons(x) = reshape(x, size(x)[1:end-1])

"""
    evaluate(::Network, state)

    (nn::Network)(state) = evaluate(nn, state)

Evaluate the neural network as an MCTS oracle on a single state.

Note, however, that evaluating state positions once at a time is slow and so you
may want to use a `BatchedOracle` along with an inference server that uses
[`evaluate_batch`](@ref).
"""
function evaluate(nn::Network, state)
  gspec = nn.gspec
  actions_mask = GI.actions_mask(GI.init(gspec, state))
  x = GI.vectorize_state(gspec, state)
  a = Float32.(actions_mask)
  xnet, anet = to_singletons.(convert_input_tuple(nn, (x, a)))
  net_output = forward_normalized(nn, xnet, anet)
  p, v, _ = from_singletons.(convert_output_tuple(net_output))
  return (p[actions_mask], v[1])
end

(nn::Network)(state) = evaluate(nn, state)

"""
    evaluate_batch(::Network, batch)

Evaluate the neural network as an MCTS oracle on a batch of states at once.

Take a list of states as input and return a list of `(P, V)` pairs as defined in the
MCTS oracle interface.
"""
function evaluate_batch(nn::Network, batch)
  gspec = nn.gspec
  X = Util.superpose((GI.vectorize_state(gspec, b) for b in batch))
  A = Util.superpose((GI.actions_mask(GI.init(gspec, b)) for b in batch))
  Xnet, Anet = convert_input_tuple(nn, (X, Float32.(A)))
  P, V, _ = convert_output_tuple(forward_normalized(nn, Xnet, Anet))
  return [(P[A[:,i],i], V[1,i]) for i in eachindex(batch)]
end

"""
    copy(::Network; on_gpu, test_mode)

A copy function that also handles CPU/GPU transfers and
test/train mode switches.
"""
function copy(network::Network; on_gpu, test_mode)
  network = Base.deepcopy(network)
  network = on_gpu ? Flux.gpu(network) : Flux.cpu(network)
  Flux.testmode!(network, test_mode)
  return network
end


######
###### Networks
######


"""
    FeedForwardHP

Hyperparameters for the simplenet architecture.

| Parameter                     | Description                                  |
|:------------------------------|:---------------------------------------------|
| `width :: Int`                | Number of neurons on each dense layer        |
| `depth_common :: Int`         | Number of dense layers in the trunk          |
| `depth_second_head = 1`             | Number of hidden layers in the actions head  |
| `depth_first_head = 1`             | Number of hidden layers in the value  head   |
| `use_batch_norm = false`      | Use batch normalization between each layer   |
| `batch_norm_momentum = 0.6f0` | Momentum of batch norm statistics updates    |
"""
@with_kw struct FeedForwardHP
  width :: Int
  depth_common :: Int # = fc_representation_layers
  depth_second_head :: Int = 1
  depth_first_head :: Int = 1
  use_batch_norm :: Bool = false
  batch_norm_momentum :: Float32 = 0.6f0
  #TODO
  encoding_size::Int64 = 10
    fc_representation_layers::Int64 = 256
    fc_dynamics_layers::Int64 = 256
    fc_reward_layers::Int64 = 256
    fc_value_layers::Int64 = 256
    fc_policy_layers::Int64 = 256
    fc_prediction_layers::Int64 = 256

end

"""
    Network
    
Any subtype `Network` must implement `Base.copy` along with
the following constructor:

    Network(game_spec, hyperparams)

The `regularized_params_` function must be overrided for all layers containing
parameters that are subject to regularization.

Subtypes are assumed to have fields
`hyper`, `gspec`, `common`, `first_head` and `second_head`. Based on those, an implementation
is provided for [`Network.forward`](@ref)
"""
@with_kw mutable struct Network
  gspec
  hyper
  downsample
  common
  first_head
  second_head
end

config=Config


function init_network(config::Config,gspec::AbstractGameSpec, hyper::Type{FeedForwardHP})
  bnmom = hyper.batch_norm_momentum
  function make_dense(indim, outdim)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      Dense(indim, outdim, relu)
    end
  end
  indim = prod(config.observation_shape) + config.stacked_observations * config.observation_shape[2] * config.observation_shape[3] #TODO
  outdim = config.encoding_size #TODO
  hsize = hyper.width
  hlayers(depth) = [make_dense(hsize, hsize) for i in 1:depth]
  common = Chain(
    flatten,
    make_dense(indim, hsize),
    hlayers(hyper.depth_common)...)
  first_head = Chain(
    hlayers(hyper.depth_first_head)...,
    Dense(hsize, 1, tanh))
    if hyper.depth_second_head
  second_head = Chain(
    hlayers(hyper.depth_second_head)...,
    Dense(hsize, outdim),
    softmax)
  return Network(gspec, hyper, common, first_head, second_head)
    else
        return Network(gspec, hyper, common, first_head)
    end
end

RepNetHP= FeedForwardHP(width= config.width, depth_common= config.fc_representation_layers, depth_first_head=1)
representation_network= init_network(config, gspec, RepNetHP)

DynNetHP= FeedForwardHP(width= config.width, depth_common= config.fc_dynamics_layers, depth_first_head=1, depth_second_head=fc_reward_layers )
dynamics_network= init_network(config, gspec, DynNetHP)

PredNetHP=FeedForwardHP(width= config.width, depth_common= config.fc_policy_layers, depth_first_head=config.fc_value_layers, depth_second_head=config.fc_policy_layers )
prediction_network= init_network(config, gspec, PredNetHP)

function initial_inference(observation)
    hidden_state= representation_network(observation)
    value, policy_logits= prediction_network(hidden_state)
    reward= zeros(some_size) #TODO
    return hidden_state, value, policy_logits, reward
end

function recurrent_inference(hidden_state,action)
    next_hidden_state, reward = dynamics_network(hidden_state, action)
    policy_logits, value = prediction_network(next_hidden_state)
    return value, reward, policy_logits, next_hidden_state
end



"""
    ResNetHP

Hyperparameters for the convolutional resnet architecture.

| Parameter                 | Type                | Default   |
|:--------------------------|:--------------------|:----------|
| `num_blocks`              | `Int`               |  -        |
| `num_filters`             | `Int`               |  -        |
| `conv_kernel_size`        | `Tuple{Int, Int}`   |  -        |
| `num_second_head_filters` | `Int`               | `2`       |
| `num_first_head_filters`  | `Int`               | `1`       |
| `batch_norm_momentum`     | `Float32`           | `0.6f0`   |

The trunk of the two-head network consists of `num_blocks` consecutive blocks.
Each block features two convolutional layers with `num_filters` filters and
with kernel size `conv_kernel_size`. Note that both kernel dimensions must be
odd.

During training, the network is evaluated in training mode on the whole
dataset to compute the loss before it is switched to test model, using
big batches. Therefore, it makes sense to use a high batch norm momentum
(put a lot of weight on the latest measurement).

# AlphaGo Zero Parameters

The network in the original paper from Deepmind features 20 blocks with 256
filters per convolutional layer.
"""
@with_kw struct ResNetHP
  num_blocks :: Int
  num_filters :: Int
  conv_kernel_size :: Tuple{Int, Int}
  num_second_head_filters :: Int = 2
  num_first_head_filters :: Int = 1
  batch_norm_momentum :: Float32 = 0.6f0
#TODO add parameters for different types like representation_network, prediction_network
  downsample::String = "resnet"
    blocks::Int64 = 16
    channels::Int64 = 256
    reduced_channels_reward::Int64 = 256
    reduced_channels_value::Int64 = 256
    reduced_channels_policy::Int64 = 256
    resnet_fc_reward_layers::Int64 = 256
    resnet_fc_value_layers::Int64 = 256
    resnet_fc_policy_layers::Int64 = 256
end

function downsample_block(size,in_channels,out_channels,bnmom)
	pad = size .÷ 2
	layers=Chain(
		Conv(size,in_channels=>out_channels÷2 ,stride=2,pad=pad),
    	[resnet_block(size,out_channels÷2, bnmom) for i in 1:2s]...,
		Conv(size, out_channels÷2=>out_channels, stride=2, pad=pad),
    	[resnet_block(size,out_channels, bnmom) for i in 1:3]...,
		MeanPool(3,stride=2,pad=1),
    	[resnet_block(size,out_channels, bnmom) for i in 1:3]...,
		MeanPool(3,stride=2,pad=1)
	)
	return layers
end

function downsample_block(in_channels,out_channels, config)
	h_w=(config.observation_shape[1]/16,config.observation_shape[2]/16)
	layers=Chain(
		Conv((h_w[1],h_w[1]), in_channels=>(in_channels+out_channels)÷2, relu, stride=4, pad=2),
		MaxPool(3, stride=2),
		Conv((5,5), (in_channels+out_channels)÷2=>out_channels, relu, pad=2),
		MaxPool(3, stride=2),
		AdaptiveMeanPool(h_w)
	)
	return layers
end

function resnet_block(size, n, bnmom)
  pad = size .÷ 2
  layers = Chain(
    Conv(size, n=>n, pad=pad),
    BatchNorm(n, relu, momentum=bnmom),
    Conv(size, n=>n, pad=pad),
    BatchNorm(n, momentum=bnmom))
  return Chain(
    SkipConnection(layers, +),
    x -> relu.(x))
end

function init_network(config::Config,gspec::AbstractGameSpec, hyper::Type{ResNetHP})
  indim = GI.state_dim(gspec)#TODO
  outdim = GI.num_actions(gspec)
  ksize = hyper.conv_kernel_size
  @assert all(ksize .% 2 .== 1)
  pad = ksize .÷ 2
  nf = hyper.num_filters
  npf = hyper.num_second_head_filters
  nvf = hyper.num_first_head_filters
  bnmom = hyper.batch_norm_momentum
	if hyper.downsample=="resnet"
		downsample=downsample_block(ksize,indim[3],outdim)#TODO outdim
	elseif hyper.downsample=="CNN"
		downsample=downsample_block(indim,outdim,config)#TODO
	else
		downsample=nothing
	end

  common = Chain(
    Conv(ksize, indim[3]=>nf, pad=pad),
    BatchNorm(nf, relu, momentum=bnmom),
    [resnet_block(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
    first_head = Chain(
        Conv((1, 1), nf=>nvf),
        BatchNorm(nvf, relu, momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * nvf, nf, relu),
        Dense(nf, 1, tanh))
        if hyper.num_second_head_filters
        second_head = Chain(
          Conv((1, 1), nf=>npf),
          BatchNorm(npf, relu, momentum=bnmom),
          flatten,
          Dense(indim[1] * indim[2] * npf, outdim),
          softmax)
		  if isnothing(downsample)
			  return Network(gspec, hyper, common, first_head, second_head)
		  else
            return Network(gspec, hyper,downsample, common, first_head, second_head)
		  end
        else
			if isnothing(downsample)
			    return Network(gspec, hyper, common, first_head)
		  	else
            	return Network(gspec, hyper,downsample, common, first_head)
		    end
        end
end

RepNetHP= ResNetHP() #TODO figure out a way to initilise these from the config file
representation_network= init_network(config, gspec, RepNetHP)

DynNetHP= ResNetHP()
dynamics_network= init_network(config, gspec, DynNetHP)

PredNetHP=ResNetHP()
prediction_network= init_network(config, gspec, PredNetHP)


end