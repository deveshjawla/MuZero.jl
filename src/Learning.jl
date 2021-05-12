module Networks

export Network, FeedForwardHP, ResNetHP, CyclicNesterov, Adam, OptimiserSpec, Network

using Parameters:@with_kw
using Statistics:mean
using CUDA
import Flux, Functors

CUDA.allowscalar(false)
array_on_gpu(::Array) = false
array_on_gpu(::CuArray) = true
array_on_gpu(arr) = error("Usupported array type: ", typeof(arr))

using Flux: relu, softmax, flatten
using Flux.Losses: mse, logitcrossentropy, crossentropy
using ParameterSchedulers: Cos,Stateful, next!

using Flux: Chain, Dense, Conv, BatchNorm, SkipConnection, MeanPool, MaxPool, AdaptiveMeanPool
import Zygote


#####
##### Support functions
#####

# Flux.@functor does not work due to Network being parametric
function Flux.functor(nn::Net) where Net <: Network
    children = (nn.common, nn.first_head, nn.second_head)
    constructor = cs -> Net(nn.hyper, cs...)
    return (children, constructor)
end

# This should be included in Flux
function lossgrads(f, args...)
    val, back = Zygote.pullback(f, args...)
    grad = back(Zygote.sensitivity(val))
    return val, grad
end

# Invert the scaling (defined in https://arxiv.org/abs/1805.11593)
invert_scaling(x) = sign(x) * ((sqrt(1 + 4 * 0.001 * (abs.(x) + 1 + 0.001))/(2 * 0.001))^2 - 1)

# Reduce the scale (defined in https://arxiv.org/abs/1805.11593)
scaling(x) = sign(x) * (sqrt(abs(x) + 1) - 1) + 0.0f0 #dims=(features, batch)

#######
####### Helper functions
#######

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

function forward(nn::Network, input)
	if isdefined(nn, nn.downsample)
		input = nn.downsample(input)
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

gc(::Network) = GC.gc(true) # CUDA.reclaim()


function forward_normalized(nn::Network, state, actions_mask)
  p, v = forward(nn, state)
  p = p .* actions_mask
  sp = sum(p, dims=1)
  p = p ./ (sp .+ eps(eltype(p)))
  p_invalid = 1 .- sp
  return (p, v, p_invalid)
end

to_singletons(x) = reshape(x, size(x)..., 1)
from_singletons(x) = reshape(x, size(x)[1:end - 1])

function evaluate(nn::Network, state)
  actions_mask = GI.actions_mask(GI.init(gspec, state))
  x = GI.vectorize_state(gspec, state)
  a = Float32.(actions_mask)
  xnet, anet = to_singletons.(convert_input_tuple(nn, (x, a)))
  net_output = forward_normalized(nn, xnet, anet)
  p, v, _ = from_singletons.(convert_output_tuple(net_output))
  return (p[actions_mask], v[1])
end

(nn::Network)(state) = evaluate(nn, state)


function evaluate_batch(nn::Network, batch)
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

@with_kw mutable struct Network
	downsample
	common
	first_head
	second_head
end

function make_dense(indim, outdim, bnmom)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      	Dense(indim, outdim, relu)
    end
end

function network(hyper::Type{FeedForwardHP})
	bnmom = hyper.batch_norm_momentum
	indim = prod(gp.observation_shape) + gp.stacked_observations * gp.observation_shape[2] * gp.observation_shape[3] # TODO
	outdim = gp.encoding_size # TODO
	hsize = hyper.width
	hlayers(depth) = [make_dense(hsize, hsize, bnmom) for _ in 1:depth]
	common = Chain(
	flatten,
	make_dense(indim, hsize, bnmom),
	hlayers(hyper.depth_common)...)
	first_head = Chain(
	hlayers(hyper.depth_first_head)...,
	Dense(hsize, 1, tanh))
	if hyper.depth_second_head
	second_head = Chain(
	hlayers(hyper.depth_second_head)...,
	Dense(hsize, outdim),
	softmax)
		return Network(common, first_head, second_head)
    else
        return Network(common, first_head)
    end
end

function initial_inference(observation)# TODO write a function which can be can actually
    hidden_state = representation_network(observation)
    value, policy_logits = prediction_network(hidden_state)
    reward = zeros(some_size) # TODO
    return hidden_state, value, policy_logits, reward
end

function recurrent_inference(hidden_state, action)
    next_hidden_state, reward = dynamics_network(hidden_state, action)
    policy_logits, value = prediction_network(next_hidden_state)
    return value, reward, policy_logits, next_hidden_state
end

function downsample_block(size, in_channels, out_channels, bnmom)
	pad = size .÷ 2
	layers = Chain(
		Conv(size, in_channels => out_channels ÷ 2, stride=2, pad=pad),
    	[resnet_block(size, out_channels ÷ 2, bnmom) for i in 1:2s]...,
		Conv(size, out_channels ÷ 2 => out_channels, stride=2, pad=pad),
    	[resnet_block(size, out_channels, bnmom) for i in 1:3]...,
		MeanPool(3, stride=2, pad=1),
    	[resnet_block(size, out_channels, bnmom) for i in 1:3]...,
		MeanPool(3, stride=2, pad=1)
	)
	return layers
end

function downsample_block(in_channels, out_channels, config)
	h_w = (gp.observation_shape[1] / 16, gp.observation_shape[2] / 16)
	layers = Chain(
		Conv((h_w[1], h_w[1]), in_channels => (in_channels + out_channels) ÷ 2, relu, stride=4, pad=2),
		MaxPool(3, stride=2),
		Conv((5, 5), (in_channels + out_channels) ÷ 2 => out_channels, relu, pad=2),
		MaxPool(3, stride=2),
		AdaptiveMeanPool(h_w)
	)
	return layers
end

function resnet_block(size, n, bnmom)
  pad = size .÷ 2
  layers = Chain(
    Conv(size, n => n, pad=pad),
    BatchNorm(n, relu, momentum=bnmom),
    Conv(size, n => n, pad=pad),
    BatchNorm(n, momentum=bnmom))
  return Chain(
    SkipConnection(layers, +),
    x -> relu.(x))
end

function network(hyper::Type{ResNetHP})
    indim = GI.state_dim(gspec)# TODO
  outdim = GI.num_actions(gspec)
  ksize = hyper.conv_kernel_size
  @assert all(ksize .% 2 .== 1)
  pad = ksize .÷ 2
  nf = hyper.num_filters
  npf = hyper.num_second_head_filters
  nvf = hyper.num_first_head_filters
  bnmom = hyper.batch_norm_momentum
	if hyper.downsample == "resnet"
		downsample = downsample_block(ksize, indim[3], outdim)# TODO outdim
	elseif hyper.downsample == "CNN"
		downsample = downsample_block(indim, outdim, config)# TODO
	else
		downsample = nothing
	end

  common = Chain(
    Conv(ksize, indim[3] => nf, pad=pad),
    BatchNorm(nf, relu, momentum=bnmom),
    [resnet_block(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
    first_head = Chain(
        Conv((1, 1), nf => nvf),
        BatchNorm(nvf, relu, momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * nvf, nf, relu),
        Dense(nf, 1, tanh))
        if hyper.num_second_head_filters
        second_head = Chain(
          Conv((1, 1), nf => npf),
          BatchNorm(npf, relu, momentum=bnmom),
          flatten,
          Dense(indim[1] * indim[2] * npf, outdim),
          softmax)
		  if isnothing(downsample)
			  return Network(common, first_head, second_head)
		  else
            return Network(downsample, common, first_head, second_head)
		  end
        else
			if isnothing(downsample)
			    return Network(common, first_head)
		  	else
            	return Network(downsample, common, first_head)
		    end
        end
end


##########
##########  Training
##########


function train!(loss, nn::Network, data, callback, tp::TrainParams)
  optimiser = Flux.ADAMW()
  schedule = Stateful(Cos(λ0=1e-4, λ1=1e-2, period=10))
  for _ in 1:tp.epochs
        optimiser.eta = next!(schedule)
    	params = Flux.params(nn)
    for (i, d) in enumerate(data)
      l, grads = lossgrads(params) do
        loss(d...)
		# TODO add L2 regularization
      end
      Flux.update!(optimiser, params, grads)
      callback(i, l)
    end
  end
end

function loss(predictions::Tuple, targets::Tuple)::Float32
	# TODO for games losses are different, add condition
	value, reward, policy_logits = predictions
	target_value, target_reward, target_policy = targets
	
    value_loss = mse(value, target_value)
    reward_loss = mse(reward, target_reward)
    policy_loss = logitcrossentropy(policy_logits, target_policy)
	# TODO with gradient batch scale the losses
    # TODO Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
	if rbp.PER
		# Correct PER bias by using importance-sampling (IS) weights
		loss *= weight_batch
	end
	# Mean over batch dimension (pseudocode do a sum)
	loss = mean(value_loss, reward_loss, policy_loss) # TODO
    return loss
end

function training(tp::TrainParams, rbp::ReplayBufferParams, checkpoint, buffer, gp::GeneralParams)
	
	# Wait for the replay buffer to be filled
    while checkpoint["num_played_games"] < 1 # TODO maybe wait until a dew games are in the buffer?
        sleep(0.1)
    end
	
	next_batch = get_batch(rbp, buffer)
	training_step = checkpoint["training_step"]
	while training_step < tp.training_steps && !checkpoint["terminate"]
		index_batch, batch = next_batch
        next_batch = get_batch(rbp, buffer)
    	observation_batch, action_batch, target_value, target_reward, target_policy, weight_batch, gradient_scale_batch = batch
		# observation_batch: Width, height, channles, batch
        # action_batch: num_unroll_steps, 1 (unsqueeze), batch
        # target_value: num_unroll_steps, batch
        # target_reward: num_unroll_steps, batch
        # target_policy: num_unroll_steps, len(action_space), batch
        # gradient_scale_batch: num_unroll_steps, batch

    	priorities = zeros(eltype(target_value), size(target_value))

		## Generate predictions
		hidden_state, value, policy_logits, reward = initial_inference(observation_batch)
		predictions = [(value, reward, policy_logits)]

		for i = 1:size(action_batch)[1]
        	value, reward, policy_logits, next_hidden_state = recurrent_inference(hidden_state, action_batch[i, :])
			# Scale the gradient by half at the start of the dynamics function (See paper appendix Training)
			next_hidden_state ./= 2.0f0
			append!(predictions, (value, reward, policy_logits))
		end
        # predictions: num_unroll_steps, 3, batch

		# Compute only value loss and policy loss for the initial_inference, put a condition in train!()
		# Assert training data type and dims for train!()
		for i in 1:length(predictions)
            value, reward, policy_logits = predictions[i]
			train!(loss, nn, data, callback, tp)
			# TODO scale the losses in the loss()
			priorities[1, :] = (abs.(value - target_value[1, :])).^rbp.PER_alpha
		end

		training_step += 1

		if rbp.PER
            # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
            update_priorities!(buffer, priorities, index_batch) # TODO
        end

		# Save to the shared storage
        if training_step % tp.checkpoint_interval == 0
            merge!(
                checkpoint,
                Dict(
					"weights" => cpu(params(model))))
            if gp.save_model
                save(checkpoint, trainer.results_path) # TODO : train!() callback is doing this?
            end
        end

        set_info!(
            checkpoint,
            Dict(
                "training_step" => training_step,
                "total_loss" => total_loss,
                "value_loss" => value_loss,
                "reward_loss" => reward_loss,
                "policy_loss" => policy_loss,
            ),
        )
	end
end
