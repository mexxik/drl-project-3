# Report

This report provides brief description of the algorithms and methods used to solve the task.

## Structure

The project consists of the following elements:
* `agent.py` - this file contains core classes that implement learning of an agent:
    * `Parameters` - container for all hyperparameters. Check a separate section about these below.
    * `OUNoise` - utility that helps to implement Ornstein-Uhlenbeck noise for better exproration.
    * `ExperienceReplay` - responsible for storage and retrieval of state-action-reward tuples used for experience replay technique.
    * `ActorNN` - PyTorch neural network with 2 hidden layers and Rectified Linear Unit (ReLU) as activation function for an actor.
    * `CriticNN` - PyTorch neural network with 2 hidden layers and Rectified Linear Unit (ReLU) as activation function for a critic.
    * `Actor` and `Critic` - classes that contain all logic for learning of agents and retrieving a policy. Check Agent and Critic section of this document for more details.
    * `AgentManager` - a class that helps to manage Agent's and Critic's functionality.
* `train.ipynb` - a Notebook for training of an agent. It contains multiple experiments, for more information check Ideas for possible future agent's performance improvement section. Some basic benchmarking, like graphs, are includes there as well.
* `test.ipynb` - a Notebook that loads trained models and renders agents playing tennis.    

## Actor and Critic

These classes are the core to learning.

This solution implemented using 1 critic and 2 actors that share experience replay buffer.

Here are the main steps of the algorithm:

* get transition tuple from the environment and store in the shared experience buffer
* where there are enough transitions available get a batch and start learning
* the actual neural network learning is implement in `learn` function of Critic class, and here are the steps:
    * step 1: getting predicted next-state actions and Q values from target models
    * step 2: computing Q targets for current states 
    * step 3: computing critic loss
    * step 4: minimizing critic loss and traing the network
    * step 5: computing actor loss (with '-' sign as this is gradient ascent
    * step 6: minimizing actor loss and traing the network
    * step 7: applying soft update and slowly mering local networks into target
* after learning is complete, soft update is applied to modify target neural network

## Hyperparameters

Here are hyperparameters used in for the training:
* BUFFER_SIZE - experience replay buffer size
* BATCH_SIZE - minibatch size to retirieve from experience replay
* GAMMA - discount rate
* TAU - parameter for soft update that determines how fast target network is merged into local
* LR_ACTOR - learning rate for actor network
* LR_CRITIC - learning rate for critic network
* WEIGHT_DECAY - L2 regularization weight decay factor
* LEARN_EVERY - determines timestep interval to learn networks
* LEARN_NUM - number of learning procedures 
* OU_SIGMA - Ornstein-Uhlenbeck noise parameter
* OU_THETA - Ornstein-Uhlenbeck noise parameter
* ENABLE_EPSILON = a flag that enables epsilon decay
* EPSILON - parameter that controls noice effect for exploration/exploitation (not enabled in this example)
* EPSILON_DECAY - decay rate for EPSILON (not enabled in this example)

## Ideas for possible future agent's performance improvement

While solving this task 6 different experiments where conducted. During each of the tries different set of hyperparameters where applied. Here are some results.

### Basic set up

The run was made with a default (basic) set of hyperparameters:
* BUFFER_SIZE = int(1e6)
* BATCH_SIZE = 128
* GAMMA = 0.99
* TAU = 2e-1
* LR_ACTOR = 1e-4
* LR_CRITIC = 1e-3
* WEIGHT_DECAY = 0
* LEARN_EVERY = 1
* LEARN_NUM = 1
* OU_SIGMA = 0.2
* OU_THETA = 0.15
* ENABLE_EPSILON = False
* EPSILON = 1.0
* EPSILON_DECAY = 1e-6

The environment was solved in 1464 episodes, and here is the graph:

![graph 1](/images/graph_01.png) 

We proceed with more experiments to find better set of hyperparameters.

### Epsilon Decay

This time `ENABLE_EPSILON` was set to `true` to enable epsilon decay.

Using this setting the environment was solved in the same 1464 episodes and this does not seem to improve performance of the learning much.


### Learn Every

Third experiment made the learning process less aggressive and initiated learning every 4 steps.

The environment was solved in 1022 episodes that is much better than the basic experiment. Also, the graph shows that the rewards were more consistent and stable.

![graph 1](/images/graph_02.png) 


### More Learning Iterations 

This time it was attempted to try learning 10 times every 20 steps.

During this experiment the environment failed to be solved in 2000 episodes.

### Batch Size Increase

The previous failed experiment was retried with increased batch size (512).

It failed to be solved as well.

### Less Aggressive with more Learning Attempts

The last experiment was conducted based on the most successful: learn 10 times every 4 steps.

Unfortunately, this one failed to solve the environment too.

## Conclusion

After a number of attempts with different set of hyperparameters it was found that the best set up is the basic one with `LEARN_EVERY` set to 4.
 

