# Optimization-of-robot-traffic

## Objective
The goal of this project is to coordinate multiple agents (robots in a warehouse) to achieve a common goal without colliding with each other. In order to do so, a CTDE (Centralised, Training, Decentralised Execution) paradigm has been implemented.

## Key Components
- **Actor** (Neural Network): Local brain of each robot. Takes a decision solely based on its own observations. (~Football player)
- **Critic** (Neural Network): Analyses the global situation and says whether the situation was great or not (~Coach)
- **Memory** (Buffer): Keeps track of what happened during a game. (~Video of the game)
- **Processing**: Mathematics module which transforms raw points into learning goals (~Statistician of the match)
- **Trainer**: Orchestrator, which makes the agents play, fills the memory and launches the training

## PPO Algorithm
PPO is a "gold standard" algorithm. Its goal is to stabilise the training. It is based on 3 mathematical ideas:
- **V**: the state-value estimate: the Critic doesn't return a score but predicts the "Return": the sum of the rewards that the agent will obtain at the end of the game.
- **A**: the advantage: the difference between what happened and what the Critic predicted. If A>0, we encourage the Actor; otherwise, we discourage him.
- **C**: the Clipping (specific to PPO): in order to avoid an abrupt change in behaviour (which could result in a crash in the training), PPO limits (clips) the importance of the update.

## Workflow
- **Data collection**: the Actors play in the environment. We store (obs, action, reward) in the memory
- **Computation of the returns**: processing.py compute the real value of each step (using \gamma)
- **Update Critic**
- **Update Actor**
- **Cleaning**
