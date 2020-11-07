# Gamblers_problem
  This script implement the dynamic programming for solving the Bellman equation for the Gambler's problem proposed in the chapter 4 of the Book  "Sutton, Richard S., and Andrew G. Barto. Reinforcement learning: An introduction. MIT press, 2018."

  The policy iteration and evaluation are implemented as a solution for the solving the Bellman equation in Markov Decision process (MDP). The problem formulation can be found in the chapter 4 of the book "Sutton, Richard S., and Andrew G. Barto. Introduction to reinforcement learning. Vol. 135. Cambridge: MIT press, 1998."

The problem is pasted here for ease of reference.

  Gambler’s Problem A gambler has the opportunity to make bets on the outcomes of a sequence of coin flips. If the coin comes up heads, he wins as many dollars as he has staked on that flip; if it is tails, he loses his stake. The game ends when the gambler wins by reaching his goal of $100, or loses by running out of money. On each flip, the gambler must decide what portion of his capital to stake, in integer numbers of dollars. This problem can be formulated as an undiscounted, episodic, finite MDP. The state is the gambler’s capital, s in {1,2,...,99} and the actions are stakes, a in {0,1,...,min(s,100-s)}. The reward is zero on all transitions except those on which the gambler reaches his goal, when it is +1. The state-value function then gives the probability of winning from each state. A policy is a mapping from levels of capital to stakes. The optimal policy maximizes the probability of reaching the goal. Let ph denote the probability of the coin coming up heads. If ph is known, then the entire problem is known and it can be solved, for instance, by value iteration. 

