""""""
"""
Template for implementing QLearner  (c) 2015 Tucker Balch

Copyright 2018, Georgia Institute of Technology (Georgia Tech)
Atlanta, Georgia 30332
All Rights Reserved

Template code for CS 4646/7646

Georgia Tech asserts copyright ownership of this template and all derivative
works, including solutions to the projects assigned in this course. Students
and other users of this template code are advised not to share it with others
or to make it available on publicly viewable websites including repositories
such as github and gitlab.  This copyright statement should not be removed
or edited.

We do grant permission to share solutions privately with non-students such
as potential employers. However, sharing with other current or future
students of CS 7646 is prohibited and subject to being investigated as a
GT honor code violation.

-----do not edit anything above this line---

Student Name: Tucker Balch (replace with your name)
GT User ID: snidadana3 (replace with your User ID)
GT ID: 903966341 (replace with your GT ID)
"""

import random as rand

import numpy as np


class QLearner(object):
    """
    This is a Q learner object.

    :param num_states: The number of states to consider.
    :type num_states: int
    :param num_actions: The number of actions available..
    :type num_actions: int
    :param alpha: The learning rate used in the update rule. Should range between 0.0 and 1.0 with 0.2 as a typical value.
    :type alpha: float
    :param gamma: The discount rate used in the update rule. Should range between 0.0 and 1.0 with 0.9 as a typical value.
    :type gamma: float
    :param rar: Random action rate: the probability of selecting a random action at each step. Should range between 0.0 (no random actions) to 1.0 (always random action) with 0.5 as a typical value.
    :type rar: float
    :param radr: Random action decay rate, after each update, rar = rar * radr. Ranges between 0.0 (immediate decay to 0) and 1.0 (no decay). Typically 0.99.
    :type radr: float
    :param dyna: The number of dyna updates for each regular update. When Dyna is used, 200 is a typical value.
    :type dyna: int
    :param verbose: If “verbose” is True, your code can print out information for debugging.
    :type verbose: bool
    """
    def __init__(
        self,
        num_states=100,
        num_actions=4,
        alpha=0.2, 
        gamma=0.9,
        rar=0.5, # Adjusted
        radr=0.99,
        dyna=0,
        verbose=False,
    ):
        """
        Constructor method
        """
        self.num_states = num_states
        self.num_actions = num_actions
        self.alpha = alpha
        self.gamma = gamma
        self.rar = rar
        self.radr = radr
        self.dyna = dyna
        self.verbose = verbose
        self.Q = np.zeros((num_states, num_actions))
        self.Dyna_T= np.zeros((0, 4), dtype=int)
        self.s = 0
        self.a = 0
        self.experience = []


    def querysetstate(self, s):
        """
        Update the state without updating the Q-table

        :param s: The new state
        :type s: int
        :return: The selected action
        :rtype: int
        """
        # self.s = s
        # action = rand.randint(0, self.num_actions - 1)
        # if self.verbose:
        #     print(f"s = {s}, a = {action}")
        # return action
        self.s = s
        
        # Decision to explore or exploit
        if rand.random() < self.rar:
            # action
            action = rand.randint(0, self.num_actions - 1)
        else:
            action = np.argmax(self.Q[s, :])

        # Update the action and decay the random action rate
        self.rar *=self.radr
        self.a = action
        
        if self.verbose:
            print(f"querysetstate: s={s}, action={action}")
        return action        

    def query(self, s_prime, r):
        """
        Update the Q table and return an action

        :param s_prime: The new state
        :type s_prime: int
        :param r: The immediate reward
        :type r: float
        :return: The selected action
        :rtype: int
        """
        # Update the Q-value for the current state-action pair
        current_q_value = self.Q[self.s, self.a]
        max_future_q = np.max(self.Q[s_prime, :])
        updated_q_value = (1 - self.alpha) * current_q_value + self.alpha * (r + self.gamma * max_future_q)
        self.Q[self.s, self.a] = updated_q_value

        # Store the experience for Dyna-Q updates
        if self.dyna > 0:
            self.experience.append((self.s, self.a, s_prime, r))
            for _ in range(self.dyna):
                # Randomly sample from past experiences
                s, a, s_next, reward = rand.choice(self.experience)
                simulated_q_value = self.Q[s, a]
                max_simulated_future_q = np.max(self.Q[s_next, :])
                self.Q[s, a] = (1 - self.alpha) * simulated_q_value + self.alpha * (reward + self.gamma * max_simulated_future_q)

        # Decide the next action
        if rand.random() < self.rar:
            next_action = rand.randint(0, self.num_actions - 1)
        else:
            next_action = np.argmax(self.Q[s_prime, :])

        # Update the state and action, and decay the random action rate
        self.s = s_prime
        self.a = next_action
        self.rar *= self.radr

        if self.verbose:
            print(f"query: s_prime={s_prime}, r={r}, action={next_action}")

        return next_action

    def author(self):
        return 'snidadana3'  # GT username

if __name__ == "__main__":
    print("Remember Q from Star Trek? Well, this isn't him")
