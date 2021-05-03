using Parameters: @with_kw


struct TrainParams
    results_path::String = "path"
    save_model::Bool = true
    training_steps::Int64 = 10000
    batch_size::Int64 = 1024
    checkpoInt_Interval::Int64 = 10
    value_loss_weight::Float64 = 0.25
    train_on_gpu::String = "has_cuda()"
end


function loss_function(
    value,
    reward,
    policy_logits,
    target_value,
    target_reward,
    target_policy,
)::Tuple #TODO
    # Cross-entropy seems to have a better convergence than MSE
    value_loss = "value vs target_value"
    reward_loss = "reward vs target_reward"
    policy_loss = "policy vs target_policy"
    return value_loss, reward_loss, policy_loss
end

function update_lr(trainer::Trainer, config::Config)
    lr =
        config.lr_init *
        config.lr_decay_rate^(trainer.training_step / config.lr_decay_steps)
    #TODO "update lr"
end

function update_weights(config::Config, trainer::Trainer, batch::Tuple)
    observation_batch,
    action_batch,
    target_value,
    target_reward,
    target_policy,
    weight_batch,
    gradient_scale_batch = batch
    # Keep values as scalars for calculating the priorities for the prioritized replay
    target_value_scalar = convert(Vector{Float32}, target_value)
    priorities = zeros(eltype(target_value_scalar), size(target_value_scalar))

    #set device
    #send observation_batch,.... all to this device #TODO

    target_value = support_to_scalar(target_value, config.support_size)
    target_reward = support_to_scalar(target_reward, config.support_size)
    ## Generate predictions
    value, reward, policy_logits, hidden_state = initial_inference(observation_batch)
    predictions = [(value, reward, policy_logits)]
    for i = 1:size(action_batch)[1]
        value, reward, policy_logits, hidden_state =
            recurrent_inference(hidden_state, action_batch[:, i])
        # Scale the gradient at the start of the dynamics function (See paper appendix Training)
        scale_gradient(hidden_state) #TODO
        append!(predictions, (value, reward, policy_logits))
    end
    ## Compute losses
    value_loss, reward_loss, policy_loss = (0, 0, 0)
    value, reward, policy_logits = predictions[1]
    # Ignore reward loss for the first batch step
    current_value_loss, _, current_policy_loss = loss_function(
        value,
        reward,
        policy_logits,
        target_value,
        target_reward,
        target_policy,
    )
    value_loss += current_value_loss
    policy_loss += current_policy_loss
    # Compute priorities for the prioritized replay (See paper appendix Training)
    pred_value_scalar = support_to_scalar(value, config.support_size)
    priorities[:, 0] = abs(pred_value_scalar - target_value_scalar[:, 0])^config.PER_alpha
    for i = 1:length(predictions)
        value, reward, policy_logits = predictions[i]
        current_value_loss, current_reward_loss, current_policy_loss = loss_function(
            value,
            reward,
            policy_logits,
            target_value,
            target_reward,
            target_policy,
        )
        # Scale gradient by the number of unroll steps (See paper appendix Training)
        scale_gradient(current_value_loss)#TODO
        scale_gradient(current_reward_loss)#TODO
        scale_gradient(current_policy_loss)#TODO
        value_loss += current_value_loss
        reward_loss += current_reward_loss
        policy_loss += current_policy_loss

        # Compute priorities for the prioritized replay (See paper appendix Training)
        pred_value_scalar = support_to_scalar(value, config.support_size)
        priorities[:, 0] =
            abs(pred_value_scalar - target_value_scalar[:, 0])^config.PER_alpha
    end
    # Scale the value loss, paper recommends by 0.25 (See paper appendix Reanalyze)
    loss = value_loss * config.value_loss_weight + reward_loss + policy_loss
    if config.PER
        # Correct PER bias by using importance-sampling (IS) weights
        loss *= weight_batch
    end
    # Mean over batch dimension (pseudocode do a sum)
    loss = mean(loss) #TODO

    # Optimize
    optimizer.zero_grad()#TODO
    backward(loss)#TODO
    optimizer.step()#TODO
    training_step += 1

    return (
        priorities,
        # For log purpose
        loss,#TODO
        mean(value_loss),#TODO
        mean(reward_loss),#TODO
        mean(policy_loss),#TODO
    )
end

function continuous_update_weights(
    rbp::ReplayBufferParams,
    config::Config,
    trainer::Trainer,
)
    # Wait for the replay buffer to be filled
    while get_info(checkpoint, "num_played_games") < 1
        sleep(0.1)
    end

    next_batch = get_batch(rbp,config, buffer)
    while trainer.training_step < config.training_steps &&
        !get_info(checkpoint, "terminate")
        index_batch, batch = next_batch
        next_batch = get_batch(rbp, config, buffer)
        update_lr(trainer, config)
        priorities, total_loss, value_loss, reward_loss, policy_loss =
            update_weights(config, trainer, batch)
        if config.PER
            # Save new priorities in the replay buffer (See https://arxiv.org/abs/1803.00933)
            update_priorities!(buffer, priorities, index_batch)
        end
        # Save to the shared storage
        if trainer.training_step % config.checkpoint_interval == 0
            set_info!(
                checkpoint,
                Dict(
                    "weights" => deepcopy(get_weights()),
                    "optimizer_state" => deepcopy(dict_to_cpu(optimizer.state_dict())),
                ),
            ) #TODO
            if config.save_model
                save_checkpoint()
            end
        end
        set_info!(
            checkpoint,
            Dict(
                "training_step" => self.training_step,
                "lr" => self.optimizer.param_groups[1]["lr"],
                "total_loss" => total_loss,
                "value_loss" => value_loss,
                "reward_loss" => reward_loss,
                "policy_loss" => policy_loss,
            ),
        )
        if config.training_delay #TODO
            sleep(config.training_delay)
        end
        if config.ratio #TODO
            while get_info(checkpoint, training_steps) /
                  get_info(checkpoint, num_played_steps) > config.ratio &&
                      get_info(checkpoint, "training_step") < config.training_steps &&
                      !get_info(checkpoint, "terminate")
                sleep(0.5)
            end
        end
    end
end