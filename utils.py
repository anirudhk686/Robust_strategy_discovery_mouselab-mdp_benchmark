import itertools as it
import numpy as np
from IPython.display import clear_output


# In[ ]:
def processState(instate):
    mask = np.ones(len(instate))
    state = np.ones(len(instate))
    for i in range(0, len(instate)):
        if hasattr(instate[i], 'sample'):
            mask[i] = -1
            state[i] = 0
        else:
            mask[i] = 1
            state[i] = instate[i]

    mask = mask[:] 


    state_arr = [
    [state[3],state[4],state[5]],
    [state[2],state[6],state[10]],
    [state[1],state[14],state[27]],
    ]

    mask_arr = [
    [mask[3],mask[4],mask[5]],
    [mask[2],mask[6],mask[10]],
    [mask[1],mask[14],mask[27]],
    ]

    out = np.stack((state_arr,mask_arr), axis=0)
    return out

# In[ ]:

def performAction(env,action):
    if action==0:
        return env.step(action)
    else:
        if action==1:
            ac_set=[1]

        if action==2:
            ac_set=[14]

        if action==3:
            ac_set=[27]

        if action==4:
            ac_set=[2,15,28]

        if action==5:
            ac_set =[6,19,32]

        if action==6:
            ac_set=[10,23,36]

        if action==7:
            ac_set=[3,7,11,16,20,24,29,33,37]

        if action==8:
            ac_set=[4,8,12,17,21,25,30,34,38]

        if action==9:
            ac_set=[5,9,13,18,22,26,31,35,39]


        for i in ac_set:
            s1, r, d, obs = env.step(i)
        if(obs==True):
            r=env.repeat_cost
        else:
            r=env.cost

        return s1,r,d,obs


# ---------- Functional utils ---------- #
from toolz.curried import *
max = curry(max)
min = curry(min)
call = lambda f: f()
@curry
def attr(name, obj):
    return getattr(obj, name)
@curry
def invoke(name, obj):
    return getattr(obj, name)()

lmap = curry(compose(list, map))
amap = curry(compose(np.array, lmap))

# ---------- Other ---------- #

def str_join(args, sep=' '):
    return sep.join(map(str, args))

def dict_product(d):
    """All possible combinations of values in lists in `d`"""
    for k, v in d.items():
        if not isinstance(v, list):
            d[k] = [v]

    for v in list(it.product(*d.values())):
        yield dict(zip(d.keys(), v))

def cum_returns(rewards):
    return np.flip(np.cumsum(np.flip(rewards, 0)), 0)

def clear_screen():
    print(chr(27) + "[2J")
    clear_output()

def softmax(x, temp=1):
    ex = np.exp((x - x.max()) / temp)
    return ex / ex.sum()

class Labeler(object):
    """Assigns unique integer labels."""
    def __init__(self, init=()):
        self._labels = {}
        self._xs = []
        for x in init:
            self.label(x)

    def label(self, x):
        if x not in self._labels:
            self._labels[x] = len(self._labels)
            self._xs.append(x)
        return self._labels[x]

    def unlabel(self, label):
        return self._xs[label]

    __call__ = label


import heapq
class PriorityQueue(list):
    def __init__(self, key, max_first=True):
        self.key = key
        self.inv = -1 if max_first else 1

    def pop(self):
        return heapq.heappop(self)[1]
        
    def push(self, item):
        heapq.heappush(self, (self.inv * self.key(item), item))
