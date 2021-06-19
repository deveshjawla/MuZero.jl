using Serialization
using Parameters:@with_kw
using Statistics:mean
using CUDA
using Flux: cpu, Parallel
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
using BSON: @save


#####
##### Support functions
#####

# This should be included in Flux
function loss_grad(f, args...)
    loss, back = Zygote.pullback(f, args...)
    grad = back(Zygote.sensitivity(loss))
    return loss, grad
end

# Invert the scaling (defined in https://arxiv.org/abs/1805.11593) of predicted_values #TODO
invert_scaling(x) = convert(Float32, sign(x) * (((sqrt(1 + 4 * 0.001 * (abs.(x) + 1 + 0.001)) - 1) / (2 * 0.001))^2 - 1))

# Reduce the scale (defined in https://arxiv.org/abs/1805.11593) of value and reward before feeding to network
scaling(x) = convert(Float32, sign(x) * (sqrt(abs(x) + 1) - 1 + 0.001*x)) # dims=(features, batch)

# function convert_input_tuple(nn, input::Tuple)
#     return map(input) do arr
#         array_on_gpu(nn.first_head[end].b) ? Flux.gpu(arr) : arr
#     end
# end

# function convert_output_tuple(output::Tuple)
# 	return map(output) do arr
# 		Flux.cpu(arr)
# 	end
# end

to_singletons(x) = reshape(x, size(x)..., 1)
squeeze(x) = reshape(x, size(x)[1:end - 1])
unsqueeze(xs::AbstractArray, dim::Integer) = reshape(xs, (size(xs)[1:dim - 1]..., 1, size(xs)[dim:end]...))

######
###### Networks
######

# custom split layer
struct Split{T}
  paths::T
end

Split(paths...) = Split(paths)

Flux.@functor Split

(m::Split)(x::AbstractArray) = map(f -> f(x), m.paths)

function make_dense(indim::Int, outdim::Int, bnmom::Float32, hyper::FeedForwardHP)
    if hyper.use_batch_norm
      Chain(
        Dense(indim, outdim),
        BatchNorm(outdim, relu, momentum=bnmom))
    else
      	Dense(indim, outdim, relu)
    end
end

hlayers(depth::Int, hsize, bnmom, hyper) = [make_dense(hsize, hsize, bnmom, hyper) for _ in 1:depth]

#######
####### FeedForward Networks
#######


function init_representation(conf::Config,hyper::FeedForwardHP)
	indim = prod([conf.observation_shape[1],conf.observation_shape[2], (conf.observation_shape[3]*(conf.stacked_observations+1)+conf.stacked_observations)])
	outdim = hyper.hidden_state_size
	bnmom = hyper.batch_norm_momentum
	hsize = hyper.width_hidden
	layers = Chain(flatten,
	make_dense(indim, hsize, bnmom, hyper),
	hlayers(hyper.depth_representation, hsize, bnmom, hyper)...,
	Dense(hsize, outdim)
	)
	return layers
end

function init_prediction(conf::Config,hyper::FeedForwardHP)
	bnmom = hyper.batch_norm_momentum
	indim = hyper.hidden_state_size
	outdim = length(conf.action_space)
	hsize = hyper.width_hidden
	common = Chain(flatten,
		make_dense(indim, hsize, bnmom, hyper),
		hlayers(hyper.depth_prediction, hsize, bnmom, hyper)...)
	value_head = Chain(
		hlayers(hyper.depth_value, hsize, bnmom, hyper)...,
		Dense(hsize, 1, tanh))
	policy_head = Chain(
		hlayers(hyper.depth_policy, hsize, bnmom, hyper)...,
		Dense(hsize, outdim),
		softmax)
	return Chain(common, Split(value_head, policy_head))
end

function init_dynamics(conf::Config,hyper::FeedForwardHP)
	bnmom = hyper.batch_norm_momentum
	indim = prod([conf.observation_shape[1],conf.observation_shape[2], (conf.observation_shape[3]+1)])
	outdim = hyper.hidden_state_size
	hsize = hyper.width_hidden
	# state_input = Chain(
	# flatten,
	# make_dense(indim, hsize, bnmom, hyper))
	# action_input = make_dense(outdim, hsize, bnmom, hyper)
	common = Chain(
		flatten,
		make_dense(indim, hsize, bnmom, hyper), 
		# Parallel(vcat, state_input, action_input),
		# make_dense(2 * hsize, hsize, bnmom, hyper),
		hlayers(hyper.depth_dynamics, hsize, bnmom, hyper)...
	)
	state_head = Chain(
		hlayers(hyper.depth_state_head, hsize, bnmom, hyper)...,
		Dense(hsize, outdim)
	)
	reward_head = Chain(
		hlayers(hyper.depth_reward, hsize, bnmom, hyper)...,
		Dense(hsize, 1, hyper.reward_activation))
	return Chain(common, Split(state_head, reward_head))
end

########
######## ResNets
########

function resnet_block(size::Tuple{Int,Int}, n::Int, bnmom::Float32)
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

function init_representation(conf::Config, hyper::ResNetHP)
	indim = conf.observation_shape
	ksize = hyper.conv_kernel_size
	@assert all(ksize .% 2 .== 1)
	pad = ksize .÷ 2
	nf = hyper.num_filters
	bnmom = hyper.batch_norm_momentum
	
	common = Chain(
		Conv(ksize, indim[3] => nf, pad=pad),
		BatchNorm(nf, relu, momentum=bnmom),
		[resnet_block(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)

	hyper.representation_output_size = Flux.outputsize(common, ((indim)..., 1))

	if downsampling
	downsample = Chain(
		Conv(size, indim[3] => indim[3], stride=2, pad=pad),
		[resnet_block(size, indim[3], bnmom) for i in 1:2]...,
		Conv(size, indim[3] => indim[3] * 2, stride=2, pad=pad),
		[resnet_block(size, indim[3] * 2, bnmom) for i in 1:3]...,
		MeanPool(3, stride=2, pad=1),
		[resnet_block(size, indim[3] * 2, bnmom) for i in 1:3]...,
		MeanPool(3, stride=2, pad=1),
		Conv(ksize, indim[3] * 2 => nf, pad=pad),
		BatchNorm(nf, relu, momentum=bnmom),
		[resnet_block(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)
		return downsample
	else
		return common
	end
end

function init_prediction(conf::Config,hyper::ResNetHP)
	indim = hyper.representation_output_size
	outdim = length(conf.action_space)
	ksize = (1, 1)
	@assert all(ksize .% 2 .== 1)
	pad = ksize .÷ 2
	nf = hyper.num_filters
	npf = hyper.num_second_head_filters
	nvf = hyper.num_first_head_filters
	bnmom = hyper.batch_norm_momentum
	hsize = hyper.width_hidden
	common = Chain(
		Conv(ksize, indim[3] => nf, pad=pad),
		BatchNorm(nf, relu, momentum=bnmom),
		[resnet_block(ksize, nf, bnmom) for i in 1:hyper.num_blocks]...)

	value_head = Chain(
        Conv(ksize, nf => nvf),
        BatchNorm(nvf, relu, momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * nvf, hsize, relu),
		hlayers(hyper.depth_value, hsize, bnmom, hyper)...,
        Dense(hsize, 1, tanh))

	policy_head = Chain(
			Conv(ksize, nf => npf),
			BatchNorm(npf, relu, momentum=bnmom),
			flatten,
			Dense(indim[1] * indim[2] * npf, hsize),
			hlayers(hyper.depth_value, hsize, bnmom, hyper)...,
			Dense(hsize, outdim),
			softmax)
	return Chain(common, Split(value_head, policy_head))
end

function init_dynamics(conf::Config,hyper::ResNetHP)
	indim = hyper.representation_output_size
	ksize = (1, 1)
	@assert all(ksize .% 2 .== 1)
	pad = ksize .÷ 2
	nvf = hyper.num_first_head_filters
	bnmom = hyper.batch_norm_momentum
	hsize = hyper.width_hidden
	common = Chain(
		Conv(ksize, indim[3] + hyper.stacked_actions => indim[3], pad=pad),
		BatchNorm(indim[3], relu, momentum=bnmom),
		[resnet_block(ksize, indim[3], bnmom) for i in 1:hyper.num_blocks]...)

	state_head = Chain(
		Conv(ksize, indim[3] => indim[3], pad=pad),
		BatchNorm(indim[3], relu, momentum=bnmom),
		[resnet_block(ksize, indim[3], bnmom) for i in 1:hyper.num_blocks]...)

	reward_head = Chain(
        Conv(ksize, indim[3] => nvf),
        BatchNorm(nvf, relu, momentum=bnmom),
        flatten,
        Dense(indim[1] * indim[2] * nvf, hsize, relu),
		hlayers(hyper.depth_value, hsize, bnmom, hyper)...,
        Dense(hsize, 1, tanh))

		return Chain(common, Split(state_head, reward_head))
end

##########
##########  Training
##########

function loss(conf, params, predictions::Tuple, targets::Tuple, weight_batch::Any, gradient_scale_batch::Matrix{Float32})::Float32
	value, reward, policy_logits = predictions
	target_values, target_rewards, target_policies = targets
	# By default we Correct PER bias by using importance-sampling (IS) weights
	# But if conf.PER is false then weight_batch is 1, and we avoid the correction
	if !conf.PER
		weight_batch = 1.0f0
	end

	# L2 regularization
	sqnorm(x) = sum(abs2, x)
	
    value_loss = mse(value, target_values, agg=x->mean((sum(x,dims=1)./gradient_scale_batch).*weight_batch))
	# TODO Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)

	if conf.intermediate_rewards
    	reward_loss = mse(reward, target_rewards, agg=x->mean((sum(x,dims=1)./gradient_scale_batch).*weight_batch))
	else
		reward_loss = 0.0f0
	end
    policy_loss = logitcrossentropy(policy_logits, target_policies, agg=x->mean((sum(x,dims=2)./gradient_scale_batch).*weight_batch))
	
	loss = sum([value_loss, reward_loss, policy_loss])
	# TODO value loss and reward loss are much smaller that the policy loss
	# merge!(
    #         progress_stats,
    #         Dict(
    #             "total_loss" => loss,
    #             "value_loss" => value_loss,
    #             "reward_loss" => reward_loss,
    #             "policy_loss" => policy_loss,
    #         ),
    #     )
	
	#loss + L2
    return loss + sum(sqnorm, params)
end

"""
Makes a state_action stack along the channels dimension, as input for the dynamics net
"""
function make_dynamics_input(states::Array{Float32,4},actions::Vector{Float32},conf::Config)::Array{Float32,4}
	actions ./= length(conf.action_space)
	state_actions=Array{Float32,4}(undef, conf.observation_shape[1], conf.observation_shape[2], conf.observation_shape[3]+1 ,0)
	for i in 1:length(actions)
		action = actions[i] * ones(Float32,(conf.observation_shape[1], conf.observation_shape[2]))
		# Scale the gradient by half at the start of the dynamics function (See paper appendix Training)
		state = states[:,:,:,i] * 2.0f0
		state_action = cat(state, action, dims=3)
		state_actions = cat(state_actions, state_action, dims=4)
	end
	return state_actions
end

function training(conf::Config, representation, prediction, dynamics, progress::Dict{String, Int}, buffer::Dict{Int,GameHistory})::Nothing
	
	# Wait for the replay buffer to be filled
    while progress["num_played_games"] < 1 # TODO, GPU should always have data available to train on.
        # @info "Waiting for replay buffer to be filled"
		sleep(0.1)
    end

	# @info "Training Started"

	optimiser = Flux.ADAMW()
  	schedule = Stateful(Cos(λ0=1e-4, λ1=1e-2, period=10))
	
	training_step = progress["training_step"]

	while training_step ≤ conf.training_steps
		training_step += 1
        next_batch = get_batch(conf, buffer)
		index_batch, batch = next_batch
    	observation_batch, action_batch, target_values, target_rewards, target_policies, weight_batch, gradient_scale_batch = batch
		gradient_scale_batch = permutedims(gradient_scale_batch)
		conf.PER ? weight_batch = permutedims(weight_batch) : nothing
		# @info "Batch Sizes are:" size(observation_batch) size(action_batch) size(target_values) size(target_rewards) size(target_policies) size(gradient_scale_batch)
		# observation_batch: Width, height, channels, batch_size
        # action_batch: num_unroll_steps + 1, batch_size
        # target_values: num_unroll_steps+1, batch_size
        # target_rewards: num_unroll_steps+1, batch_size
        # target_policies: len(action_space), num_unroll_steps+1, batch_size
        # weight_batch, gradient_scale_batch: 1, batch_size

    	priorities = zeros(eltype(target_values), size(target_values))

		# # Laod the latest saved networks
		# if training_step % conf.checkpoint_interval == 0 && progress["training_step"]>1
		# 	representation= deserialize(joinpath(conf.networks_path,"$(training_step)_representation.bin"))
		# 	prediction= deserialize(joinpath(conf.networks_path,"$(training_step)_prediction.bin"))
		# 	dynamics= deserialize(joinpath(conf.networks_path,"$(training_step)_dynamics.bin"))
		# 	@info "Latest Networks successfully reloaded during training"
        # end

		## Generate predictions, first for the observation then for num_unroll_steps*hidden_states
		hidden_state = representation(observation_batch)
		if ndims(hidden_state)==2
			hidden_state=reshape(hidden_state, (conf.observation_shape...,conf.batch_size))
		end
		predicted_values, predicted_policies = prediction(hidden_state)
		predicted_rewards = zeros((1,conf.batch_size))
		predicted_policies=Flux.unsqueeze(predicted_policies, 2)

		for i = 1:conf.num_unroll_steps
        	value, policy_logits = prediction(hidden_state)
			if ndims(hidden_state)==2
				hidden_state=reshape(hidden_state, (conf.observation_shape...,conf.batch_size))
			end
			policy_logits=Flux.unsqueeze(policy_logits, 2)
			state_action=make_dynamics_input(hidden_state, action_batch[i,:],conf)
			hidden_state, reward = dynamics(state_action)
			if ndims(hidden_state)==2
				hidden_state=reshape(hidden_state, (conf.observation_shape...,conf.batch_size))
			end
			# @info "Size of Predictions" size(value) size(reward) size(policy_logits) size(predicted_values) size(predicted_rewards) size(predicted_policies)
			predicted_values= vcat(predicted_values, value)
			predicted_rewards= vcat(predicted_rewards, reward)
			predicted_policies= cat(predicted_policies, policy_logits, dims=2)
		end

        # predictions & targets: if_any_other_dim, num_unroll_steps + 1, batch
		targets = (target_values, target_rewards, target_policies)
		predictions = (predicted_values, predicted_rewards, predicted_policies)

		# @info "Predictions and Targets generated successfully" size(target_values) size(target_rewards) size(target_policies) size(predicted_values) size(predicted_rewards) size(predicted_policies)

		params_representation = Flux.params(representation)
		params_prediction = Flux.params(prediction)
		params_dynamics = Flux.params(dynamics)

		optimiser[1].eta = next!(schedule) # changes with every minibatch

		# calculates losses over all the samples in this batch at once
		l_representation, grads_representation = loss_grad(params_representation) do
			loss(conf, params_representation, predictions, targets, weight_batch, gradient_scale_batch) 
		end
		l_prediction, grads_prediction = loss_grad(params_prediction) do
			loss(conf, params_prediction, predictions, targets, weight_batch, gradient_scale_batch) 
		end
		l_dynamics, grads_dynamics = loss_grad(params_dynamics) do
			loss(conf, params_dynamics, predictions, targets, weight_batch, gradient_scale_batch) 
		end

		@info "Representation loss =" l_representation
		@info "Prediction loss =" l_prediction
		@info "Dynamics loss=" l_dynamics

		Flux.update!(optimiser, params_representation, grads_representation)
		Flux.update!(optimiser, params_prediction, grads_prediction)
		Flux.update!(optimiser, params_dynamics, grads_dynamics)

		priorities = (abs.(predicted_values - target_values)).^conf.PER_alpha

		
		# @info "Training progress at" training_step
		# println(progress)
		
		if conf.PER
            # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
            update_priorities!(buffer, priorities, index_batch)
        end
		
		# Save to the shared storage(disk) #TODO make them availbal on memory
        if training_step % conf.checkpoint_interval == 0 && training_step > 1 
			representation= cpu(representation)
			prediction= cpu(prediction)
			dynamics= cpu(dynamics)
			serialize(joinpath(conf.networks_path,"$(training_step)_representation.bin"), representation)
			serialize(joinpath(conf.networks_path,"$(training_step)_prediction.bin"), prediction)
			serialize(joinpath(conf.networks_path,"$(training_step)_dynamics.bin"), dynamics)
        end
		progress["training_step"] = training_step
	end
	return nothing
end
