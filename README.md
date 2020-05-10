# Robust strategy discovery benchmark problems

This repository contains the code for generating the benchmark environments for Robust strategy discovery in a 333 environment setup in mouselab mdp paradigm. The details of the environment structure is explained in our CogSci paper(published in 2020, link soon to be updated).

This code helps to generate train and test environments as per the prior and likelihoods explained in our paper. Environments are of the mouselab-mdp type [Mouselab-MDP: A new paradigm for tracing how people plan](https://osf.io/7wcya). They are gym environments and hence suitable for RL algorithms.

The action space of the environment could be understood using the below image:

![Image of action_space](https://github.com/anirudhk686/Robust_strategy_discovery_mouselab-mdp_benchmark/blob/master/action_space.png)

Each action is just a number and it uncovers the node as shown above. Note '0' is used to terminate the episode if you do not want to uncover more nodes and move the plane.

Note that in the Mouselab object the environment and its state is stored as a tree that might be difficult to operate upon, especially when using neural nets so please use the 2 functions present in utils.py:

1) processState(env.state) : this takes in the state given of the mouselabmdp object and gives out 2d matrix representation of the state with 2 layers. First layer is a 2d matrix of rewards (uncovered node reward is assumed as zero). Second layer is a 2d matrix that is a mask array which tells which of the nodes are clicked/unclicked (1 : clicked, -1 : unclicked)
2) performAction(env,action_number) : Instead using env.step() directly, please use this function to perform action. It returns the next state, reward, is_done, is_observed(in case you click on already observed node).


To generate the benchmark environments:

Step 1 : first learn a posterior function( using a neural net, refer paper for details) and store the model. please run posterior.py for this.\
Step 2 : Now as described in the paper with the prior and likelihood considered, there are 36 possiblities of theta's and 6 theta-hats. 
a) to get a particular instance of theta-hat use : get_eval_that(int type_range_0to5) in the gen_environments.py.\
b) to get the train environments given a particular theta-hat : get_train_envs(int type_range_0to5) in the gen_environments.py.(returns around 4000 environments as per the posterior and the posterior array over the 36 theta)\
c) to get the test environments given a particular theta-hat : get_test_envs(int type_range_0to5) in the gen_environments.py.(returns around 1000 environments as per the posterior and the posterior array over the 36 theta)

To calculate the benchmark score:

a) For each of the theta-hat type 0 to 5:
- train your algorithm using the train_envs.
- calculate the average reward per environment in test_envs for that theta-hat type (avg_reward_theta-hat)

b) Compute average over the 6 avg_reward_theta-hat's 










