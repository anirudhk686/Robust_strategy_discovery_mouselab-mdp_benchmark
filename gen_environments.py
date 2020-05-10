# In[ ]:

from posterior import *


# In[ ]:
def get_eval_that(type=0):
    that_probs = [[0,0,0],[1,1,1],[2,2,2],[3,3,3],[4,4,4],[5,5,5]]
    that = that_probs[type]

    outcomes_probs = {'h':[0.25,0.25,0,0,0,0,0,0,0,0,0.25,0.25],'m':[0,0,0.25,0.25,0,0,0,0,0.25,0.25,0,0],'l':[0,0,0,0,0.25,0.25,0.25,0.25,0,0,0,0]}

    level_outcomes=[
    outcomes_probs['l']+outcomes_probs['m']+outcomes_probs['h'],
    outcomes_probs['l']+outcomes_probs['h']+outcomes_probs['m'],
    outcomes_probs['m']+outcomes_probs['l']+outcomes_probs['h'],
    outcomes_probs['m']+outcomes_probs['h']+outcomes_probs['l'],
    outcomes_probs['h']+outcomes_probs['l']+outcomes_probs['m'],
    outcomes_probs['h']+outcomes_probs['m']+outcomes_probs['l'],
    ]

    that = np.array([level_outcomes[that[0]],level_outcomes[that[1]],level_outcomes[that[2]]]).reshape((1,3,36,1))
    return that


# In[ ]:
def get_train_envs(theta_hat_num,repeat_cost=3,num_envs=4000):
    pos = get_posterior(get_eval_that(theta_hat_num))
    pos1 = np.array(pos[:36])
    pos2 = np.array(pos[36:])

    theta_space = [
    [0,0,0],[0,1,1],[0,2,2],[0,3,3],[0,4,4],[0,5,5],
    [1,0,0],[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5],
    [2,0,0],[2,1,1],[2,2,2],[2,3,3],[2,4,4],[2,5,5],
    [3,0,0],[3,1,1],[3,2,2],[3,3,3],[3,4,4],[3,5,5],
    [4,0,0],[4,1,1],[4,2,2],[4,3,3],[4,4,4],[4,5,5],
    [5,0,0],[5,1,1],[5,2,2],[5,3,3],[5,4,4],[5,5,5],
    ]


    var = {'h':[-48,-24,24,48],'m':[-8,-4,4,8],'l':[-2,-1,1,2]}
    level_types=[
    [Categorical(var['l']),Categorical(var['m']),Categorical(var['h'])],
    [Categorical(var['l']),Categorical(var['h']),Categorical(var['m'])],
    [Categorical(var['m']),Categorical(var['l']),Categorical(var['h'])],
    [Categorical(var['m']),Categorical(var['h']),Categorical(var['l'])],
    [Categorical(var['h']),Categorical(var['l']),Categorical(var['m'])],
    [Categorical(var['h']),Categorical(var['m']),Categorical(var['l'])],
    ]

    reward_arr=[]
    for temp in theta_space:
        tarray = [level_types[temp[2]],level_types[temp[1]],level_types[temp[0]]]
        reward_arr.append(tarray)



    env_array=[]
    branching = [3, 3, 3]
    cost=1


    num_arr=[]

    pos=pos1
    pmodel_num=-1
    j=0

    for rarray in reward_arr:
        tnum = int(pos[j]*num_envs)
        num_arr.append(tnum)
        tenv = [MouselabEnv.new_symmetric(branching, rarray, seed=j, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost,process_model=pmodel_num) for j in range(tnum)]
        env_array = env_array+tenv
        j=j+1


    pos=pos2
    pmodel_num=1
    j=0
    for rarray in reward_arr:
        tnum = int(pos[j]*num_envs)
        num_arr.append(tnum)
        tenv = [MouselabEnv.new_symmetric(branching, rarray, seed=j, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost,process_model=pmodel_num) for j in range(tnum)]
        env_array = env_array+tenv
        j=j+1



    pos=pos1+pos2

    return env_array,pos
# In[ ]:


def get_test_envs(theta_hat_num,repeat_cost=0,num_envs=1000):
    pos = get_posterior(get_eval_that(theta_hat_num))
    pos1 = np.array(pos[:36])
    pos2 = np.array(pos[36:])

    theta_space = [
    [0,0,0],[0,1,1],[0,2,2],[0,3,3],[0,4,4],[0,5,5],
    [1,0,0],[1,1,1],[1,2,2],[1,3,3],[1,4,4],[1,5,5],
    [2,0,0],[2,1,1],[2,2,2],[2,3,3],[2,4,4],[2,5,5],
    [3,0,0],[3,1,1],[3,2,2],[3,3,3],[3,4,4],[3,5,5],
    [4,0,0],[4,1,1],[4,2,2],[4,3,3],[4,4,4],[4,5,5],
    [5,0,0],[5,1,1],[5,2,2],[5,3,3],[5,4,4],[5,5,5],
    ]


    var = {'h':[-48,-24,24,48],'m':[-8,-4,4,8],'l':[-2,-1,1,2]}
    level_types=[
    [Categorical(var['l']),Categorical(var['m']),Categorical(var['h'])],
    [Categorical(var['l']),Categorical(var['h']),Categorical(var['m'])],
    [Categorical(var['m']),Categorical(var['l']),Categorical(var['h'])],
    [Categorical(var['m']),Categorical(var['h']),Categorical(var['l'])],
    [Categorical(var['h']),Categorical(var['l']),Categorical(var['m'])],
    [Categorical(var['h']),Categorical(var['m']),Categorical(var['l'])],
    ]

    reward_arr=[]
    for temp in theta_space:
        tarray = [level_types[temp[2]],level_types[temp[1]],level_types[temp[0]]]
        reward_arr.append(tarray)



    env_array=[]
    branching = [3, 3, 3]
    cost=1


    num_arr=[]

    pos=pos1
    pmodel_num=-1
    j=0

    for rarray in reward_arr:
        tnum = int(pos[j]*num_envs)
        num_arr.append(tnum)
        tenv = [MouselabEnv.new_symmetric(branching, rarray, seed=j+4000, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost,process_model=pmodel_num) for j in range(tnum)]
        env_array = env_array+tenv
        j=j+1


    pos=pos2
    pmodel_num=1
    j=0
    for rarray in reward_arr:
        tnum = int(pos[j]*num_envs)
        num_arr.append(tnum)
        tenv = [MouselabEnv.new_symmetric(branching, rarray, seed=j+4000, cost=cost, sample_term_reward=False, env_type='new',repeat_cost=repeat_cost,process_model=pmodel_num) for j in range(tnum)]
        env_array = env_array+tenv
        j=j+1




    pos=pos1+pos2

    return env_array,pos
# In[ ]:


