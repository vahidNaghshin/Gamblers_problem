#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
This script implement the dynamic programming for solving the Bellman equation for the Gambler's problem
proposed in the chapter 4 of the Book 
"Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018."
"""

__author__ = "Vahid Naghshin"
__email__ = "vnagh@dolby.com"

import numpy as np
from matplotlib import pylab as plt


# actions = {0, 1, ..., min(100-s, s)}. Each state has its own action set.
def state_action_table(p_head, max_state):
    '''
    This func derives the action-state-reward table
    '''
    state_dic = {}
    for state in range(0, max_state + 2):
        action_set = {}
        if state > 0 and state <= max_state:
            for action in range(len(np.arange(0, min(state, max_state-state+1)+1))):
                if action:
                    action_set[action] = [(p_head,
                                           min(max_state+1,
                                               action+state),
                                           1 if min(max_state+1,
                                                    action+state) == max_state+1
                                           else 0,
                                           True if min(max_state+1,
                                                       action+state) == max_state+1
                                           else False),
                                          (1 - p_head,
                                           state - action,
                                           0,
                                           True if state-action == 0 else False)]
                else:
                    action_set[action] = [(1, state, 0, False)]

            state_dic[state] = action_set
        else:
            state_dic[state] = {0: [(1, state, 0, True)]}

    return state_dic


def policy_iteration(env, gamma=1.0):
    random_policy = np.zeros(env.observation_space.n)
    no_of_iterations = 200000
    for i in range(no_of_iterations):
        new_value_function = compute_value_function(random_policy, gamma)
        new_policy = extract_policy(new_value_function, gamma)
        if (np.all(random_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i+1))
            break
        random_policy = new_policy
    return new_policy


def value_iteration(env, max_state, gamma=1.0, no_of_iterations=3):
    value_table = np.zeros(max_state + 2)
    threshold = 0.01
    for i in range(no_of_iterations):
        updated_value_table = np.copy(value_table)
        for state in range(1, max_state+1):
            Q_value = []
            for action in env[state].keys():
                next_states_rewards = []
                for next_sr in env[state][action]:
                    trans_prob, next_state, reward_prob, _ = next_sr
                    next_states_rewards.append((trans_prob *
                                                (reward_prob +
                                                 gamma *
                                                 updated_value_table[next_state])))
                Q_value.append(np.sum(next_states_rewards))
            value_table[state] = max(Q_value)
        if (np.sum(np.fabs(updated_value_table - value_table)) <=
                threshold):
            print('Value-iteration converged at iteration# %d.' % (i+1))
            break
    return value_table


def extract_policy(env, value_table, max_state, gamma=1.0):
    policy = np.zeros(max_state+2)
    for state in range(1, max_state + 2):
        Q_table = np.zeros(len(env[state].keys()))
        for action in env[state].keys():
            for next_sr in env[state][action]:
                trans_prob, next_state, reward_prob, _ = next_sr
                Q_table[action] += (trans_prob * (reward_prob +
                                                  gamma * value_table[next_state]))
        policy[state] = np.where(Q_table == np.max(Q_table))[0][-1]
    return policy


#  In this phase, given a policy (set of actions) we evaluate the policy, i.e.,
# calculate the value-function for each state
def compute_value_function(policy, env, value_table, max_state=99, gamma=1.0):
    #     value_table = np.zeros(max_state+2)
    threshold = 1e-10
    while True:
        updated_value_table = np.copy(value_table)
        for state in range(1, max_state + 2):
            action = policy[state]
            value_table[state] = sum([trans_prob * (reward_prob + gamma *
                                                    updated_value_table[next_state])
                                      for trans_prob, next_state, reward_prob, _ in
                                      env[state][action]])

        if (np.sum((np.fabs(updated_value_table - value_table))) <= threshold):
            break
    return value_table


def policy_iteration(env, max_state, gamma=1.0, no_of_iterations=10000):
    value_table = value_iteration(env, max_state, gamma, no_of_iterations)
    random_policy = np.zeros(max_state + 2)
    for i in range(no_of_iterations):
        new_policy = extract_policy(env, value_table, max_state, gamma=1.0)
        value_table = compute_value_function(new_policy, env, value_table,
                                             max_state=99, gamma=1.0)
        if (np.all(random_policy == new_policy)):
            print('Policy-Iteration converged at step %d.' % (i+1))
            break
        random_policy = new_policy

    return random_policy, value_table


def main():

    p_head = 0.4  # prob. of head
    max_state = 99

    # the state-action-reward tuple is construcred. The status of game
    # is also included.
    env = state_action_table(p_head, max_state)

    random_policy, value_table = policy_iteration(env, max_state,
                                                  gamma=1.0,
                                                  no_of_iterations=10000)

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, max_state+1), value_table[1:-1])
    plt.xlabel("states", fontsize=22)
    plt.ylabel("Value Estimate", fontsize=22)
    plt.show()

    plt.figure(figsize=(10, 7))
    plt.plot(np.arange(1, max_state+1), random_policy[1:-1])
    plt.xlabel("states", fontsize=22)
    plt.ylabel("Final Policy (stake)", fontsize=22)
    plt.show()


if __name__ == '__main__':
    main()
