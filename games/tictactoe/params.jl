
const conf = Config(
	observation_shape = (3,3,3),
	action_space = collect(1:9),
	players = collect(1:2),
	stacked_observations = 1,
	num_workers = 2,
	max_moves = 9,
	num_unroll_steps = 5,
	td_steps = 5,
	PER = false,
	opponent= "human",
	training_steps = 10000,
	batch_size = 32,
	num_iters = 10
	)

const hyper = FeedForwardHP(
	width_hidden = 64,
	depth_representation = 3,
	depth_prediction = 3,
	depth_dynamics = 3,
	depth_policy = 1,
	depth_value = 1,
	depth_reward = 1,
	depth_state_head = 3,
	hidden_state_size = 27,
	reward_activation = tanh
)