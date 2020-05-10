# In[ ]:
import os
import numpy as np
import tensorflow as tf
import tensorflow.contrib.slim as slim
from mouselab import MouselabEnv
from distributions import Categorical, Normal
from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Conv2D, Flatten, Dropout
from sklearn.utils import shuffle
from itertools import combinations_with_replacement, permutations


# In[ ]:
def get_train_dataP1(num_samples=3000):

    temperature = .3

    theta_space = [
        [0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4], [0, 5, 5],
        [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 5, 5],
        [2, 0, 0], [2, 1, 1], [2, 2, 2], [2, 3, 3], [2, 4, 4], [2, 5, 5],
        [3, 0, 0], [3, 1, 1], [3, 2, 2], [3, 3, 3], [3, 4, 4], [3, 5, 5],
        [4, 0, 0], [4, 1, 1], [4, 2, 2], [4, 3, 3], [4, 4, 4], [4, 5, 5],
        [5, 0, 0], [5, 1, 1], [5, 2, 2], [5, 3, 3], [5, 4, 4], [5, 5, 5],
    ]

    one_hot_vectors = np.eye(72, dtype=int)

    all_theta = np.empty((0, 72))
    all_that = np.empty((0, 3, 36, 1))
    for y in range(len(theta_space)):

        theta = theta_space[y]

        combi = list(combinations_with_replacement(theta, r=3))
        all = []
        for i in combi:
            all = all + list(permutations(i))
        possiblities = list(set(all))

        num_main = []
        for pos in possiblities:
            num = []
            temp0 = []
            for i in range(3):
                if pos[0] == theta[i]:
                    temp0.append(i)

            for i in temp0:
                temp1 = []
                for j in range(3):
                    if pos[1] == theta[j]:
                        temp1.append(j)

                for j in temp1:
                    temp2 = []
                    for k in range(3):
                        if pos[2] == theta[k]:
                            temp2.append(k)

                    for k in temp2:
                        sum = (i - 0) + (j - 1) + (k - 2)
                        num.append(sum)

            num_main.append(num)

        exp_array = []
        for arr in num_main:
            sum = 0

            for i in arr:
                sum = sum + np.exp(-(i / temperature))
            exp_array.append(sum)

        exp_array = np.array(exp_array)

        that_probs = exp_array / exp_array.sum()
        that = possiblities

        outcomes_probs = {'h': [0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25], 'm': [
            0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0, 0], 'l': [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]}

        level_outcomes = [
            outcomes_probs['l'] + outcomes_probs['m'] + outcomes_probs['h'],
            outcomes_probs['l'] + outcomes_probs['h'] + outcomes_probs['m'],
            outcomes_probs['m'] + outcomes_probs['l'] + outcomes_probs['h'],
            outcomes_probs['m'] + outcomes_probs['h'] + outcomes_probs['l'],
            outcomes_probs['h'] + outcomes_probs['l'] + outcomes_probs['m'],
            outcomes_probs['h'] + outcomes_probs['m'] + outcomes_probs['l'],
        ]

        temp_that = np.empty((0, 3, 36, 1))
        for i in range(len(possiblities)):
            tarr = possiblities[i]
            tnum = int(that_probs[i] * num_samples)
            if(tnum == 0):
                continue
            z = np.array([level_outcomes[tarr[0]], level_outcomes[tarr[1]],
                          level_outcomes[tarr[2]]]).reshape((3, 36, 1))
            temp_that = np.append([z] * tnum, temp_that, axis=0)

        temp_theta = np.array([list(one_hot_vectors[y])] * temp_that.shape[0])

        all_that = np.append(temp_that, all_that, axis=0)
        all_theta = np.append(temp_theta, all_theta, axis=0)

    all_theta, all_that = shuffle(all_theta, all_that)

    pmodel_one_hot = np.eye(2, dtype=int)
    pmodel = np.array([list(pmodel_one_hot[0])] * all_that.shape[0])

    return all_theta, all_that

# In[ ]:


def get_train_dataP2(num_samples=3000):

    temperature = .3

    theta_space = [
        [0, 0, 0], [0, 1, 1], [0, 2, 2], [0, 3, 3], [0, 4, 4], [0, 5, 5],
        [1, 0, 0], [1, 1, 1], [1, 2, 2], [1, 3, 3], [1, 4, 4], [1, 5, 5],
        [2, 0, 0], [2, 1, 1], [2, 2, 2], [2, 3, 3], [2, 4, 4], [2, 5, 5],
        [3, 0, 0], [3, 1, 1], [3, 2, 2], [3, 3, 3], [3, 4, 4], [3, 5, 5],
        [4, 0, 0], [4, 1, 1], [4, 2, 2], [4, 3, 3], [4, 4, 4], [4, 5, 5],
        [5, 0, 0], [5, 1, 1], [5, 2, 2], [5, 3, 3], [5, 4, 4], [5, 5, 5],
    ]

    one_hot_vectors = np.eye(72, dtype=int)

    all_theta = np.empty((0, 72))
    all_that = np.empty((0, 3, 36, 1))
    for y in range(len(theta_space)):

        temp_theta = theta_space[y]

        tarr = [temp_theta[1], temp_theta[1], temp_theta[1]]

        outcomes_probs = {'h': [0.25, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.25], 'm': [
            0, 0, 0.25, 0.25, 0, 0, 0, 0, 0.25, 0.25, 0, 0], 'l': [0, 0, 0, 0, 0.25, 0.25, 0.25, 0.25, 0, 0, 0, 0]}

        level_outcomes = [
            outcomes_probs['l'] + outcomes_probs['m'] + outcomes_probs['h'],
            outcomes_probs['l'] + outcomes_probs['h'] + outcomes_probs['m'],
            outcomes_probs['m'] + outcomes_probs['l'] + outcomes_probs['h'],
            outcomes_probs['m'] + outcomes_probs['h'] + outcomes_probs['l'],
            outcomes_probs['h'] + outcomes_probs['l'] + outcomes_probs['m'],
            outcomes_probs['h'] + outcomes_probs['m'] + outcomes_probs['l'],
        ]

        temp_that = np.array([level_outcomes[tarr[0]], level_outcomes[tarr[1]],
                              level_outcomes[tarr[2]]]).reshape((3, 36, 1))
        temp_that = np.array([temp_that] * num_samples)

        temp_theta = np.array(
            [list(one_hot_vectors[y + 36])] * temp_that.shape[0])

        all_that = np.append(temp_that, all_that, axis=0)
        all_theta = np.append(temp_theta, all_theta, axis=0)

    all_theta, all_that = shuffle(all_theta, all_that)

    pmodel_one_hot = np.eye(2, dtype=int)
    pmodel = np.array([list(pmodel_one_hot[1])] * all_that.shape[0])

    return all_theta, all_that

# In[ ]:


def posterior_train_data():
    labels1, data1 = get_train_dataP1()
    labels2, data2 = get_train_dataP2()

    data = np.append(data1, data2, axis=0)
    labels = np.append(labels1, labels2, axis=0)

    data, labels = shuffle(data, labels)
    return labels, data

# In[ ]:


def train_posterior_function():



    path = 'posterior/model'

    def build_model():
        model = Sequential()
        model.add(Conv2D(64, kernel_size=(1, 12), strides=(
            1, 12), activation='relu', input_shape=(3, 36, 1)))
        model.add(Conv2D(64, (1, 3), activation='relu'))
        model.add(Conv2D(64, (3, 1), activation='relu'))
        model.add(Flatten())
        model.add(Dense(128, activation='relu'))
        model.add(Dropout(0.5))
        model.add(Dense(72, activation='softmax'))

        return model

    model = build_model()

    model.compile(
        optimizer='adam',
        loss='categorical_crossentropy',
    )

    labels, data = posterior_train_data()

    model.fit(
        data,  # training data
        labels,  # training targets
        epochs=5,
        batch_size=32,
    )

    model.save(path + '.h5')



# In[ ]:


def get_posterior(theta_hat):
    path = 'posterior/model' + '.h5'
    model = tf.keras.models.load_model(path)
    posterior = model.predict(theta_hat)[0]

    return posterior


# In[ ]:
if __name__ == '__main__':
    os.makedirs('./posterior')
    train_posterior_function()
