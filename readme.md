# MuZero.jl

This package provides the core MuZero algorithm in Julia Language:

[MuZero](https://arxiv.org/abs/1911.08265) is a state of the art RL algorithm for board games (Chess, Go, ...) and Atari games.
It is the successor to [AlphaZero](https://arxiv.org/abs/1712.01815) but without any knowledge of the environment underlying dynamics. MuZero learns a model of the environment and uses an internal representation that contains only the useful information for predicting the reward, value, policy and transitions.

Because MuZero is resource-hungry, the motivation for this project is to provide an implementation of
MuZero that is simple enough to be widely accessible, while also being
sufficiently powerful and fast to enable meaningful experiments on limited
computing resources.
I found the [Julia language](https://julialang.org/) to be instrumental in achieving this goal.

## Training a TicTacToe Agent

To download MuZero.jl and start training a TicTacToe agent with 2 threads, just run:

```sh
git clone https://github.com/deveshjawla/MuZero.jl
cd MuZero.jl
julia --project -e 'import Pkg; Pkg.instantiate()'
julia --project -t 3 ./games/tictactoe/main.jl
```

Note that the MuZero agent is not exposed to the baselines during training and
learns purely from self-play, without any form of supervision or prior knowledge.

## Features

* Residual Network and Fully connected network in [Flux](https://github.com/FluxML/Flux.jl)
* Reinforcement Learning enviornment and TicTacToe example adapted from [ReinforcementLearning.jl](https://github.com/JuliaReinforcementLearning/ReinforcementLearning.jl)
* Parallel computing natively supported by Julia
* Multi GPU support for the training and the selfplay
* Model weights automatically saved at checkpoints
* Single and two player mode
* Easily adaptable for new games

## Games implemented

* Tic-tac-toe   (Tested with the fully connected network)

### Config

You can adapt the configurations of each game by editing the `Config` of the `params.jl` file in the [games folder](https://github.com/deveshjawla/MuZero.jl/tree/master/games).

## Contribution Guide

I would like to invite you to contribute to this project by addressing any of the following points:
* _User Interface_: Session management, track Learning Performance with TensorBoard, and Diagnostic tools to understand the learned model
* _Benchmarking_: Interface and tools for Benchmarking against Perfect solvers, MCTS Only or Network Only players.
* _Logging Tools_: To track code performance
* _Optimize code for Performance_
* _Support for more than 2 Players_
* _Hyper-Parameter Search_
* _Support for Continuous action spaces_
* _Support of New environments_: Zero sum games, RL, Control problems etc.

The next aim for me would be to implement an easy to use Interface and this could be expected in v0.4.0. The __User Interface and Benchmarking__ will most likely be adapted from Jonathan Laurent's [AlphaZero.jl](https://github.com/jonathan-laurent/AlphaZero.jl).

## Acknowledgements and Citation

1. __David Foster__ for his excellent [tutorial](https://medium.com/applied-data-science/how-to-build-your-own-muzero-in-python-f77d5718061a)
2. __Werner Duvaud__ : the core algorithm of this Julia implementation is mostly based on his Python implementation. Some parts of this `ReadMe` are also adpated from his [Github repository](https://github.com/werner-duvaud/muzero-general)
3. __Julian Schrittweiser__ for his [tutorial](https://www.furidamu.org/blog/2020/12/22/muzero-intuition/) and the associated [pseudocode](https://arxiv.org/src/1911.08265v2/anc/pseudocode.py)
4. __Jonathan Laurent__ : Some parts of this `ReadMe` are adpated from his [Github repository](https://github.com/jonathan-laurent/AlphaZero.jl)

## Authors and Contributors

* Author: Devesh Jawla
* [Contributors](https://github.com/deveshjawla/MuZero.jl/graphs/contributors)

## Supporting and Citing

If you want to support this project and help it gain visibility, please consider starring
the repository. Doing well on such metrics may also help us secure academic funding in the
future. Also, if you use this software as part of your research, I would appreciate that
you include the following [citation](./CITATION.bib) in your paper.
