# Robust_strategy_discovery_mouselab_benchmark

This repository contains the code for generating the benchmark environments for Robust strategy discovery in a 333 environment setup in mouselab mdp paradigm. The details of the environment structure is explained in our CogSci paper(published in 2020, link soon to be updated).

This code helps to generate train and test environments as per the prior and likelihoods explained in our paper. Environments are of the mouselab-mdp type [Mouselab-MDP: A new paradigm for tracing how people plan](https://osf.io/7wcya). They are gym environments and hence suitable for RL algorithms.

The action space of the environment could be understood using the below image:

![Image of action_space](https://github.com/anirudhk686/Robust_strategy_discovery_mouselab-mdp_benchmark/blob/master/action_space.png)

Each action is just a number and it uncovers the node as shown above. Note '0' is used to terminate the episode if you do not want to uncover more nodes and move the plane.

Note that in the Mouselab object the environment and its state is stored as a tree that might be difficult to operate upon, especially when using neural nets so please use the 2 functions present in utils.py:
1) processState(env.state) : this takes in the state given of the mouselabmdp object and gives out 2d matrix representation of the state with 2 layers. First layer is a 2d matrix of rewards (uncovered node reward is assumed as zero). Second layer is a 2d matrix that is a mask array which tells which of the nodes are clicked/unclicked (1 : clicked, -1 : unclicked)
2) performAction(env,action_number) : Instead using env.step() directly, please use this function to perform action. It returns the next state, reward, is_done, is_observed(in case you click on already observed node).
