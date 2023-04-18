# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
# 
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util

from learningAgents import ValueEstimationAgent
import collections

class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        "*** YOUR CODE HERE ***"

        state_list = self.mdp.getStates()

        # Start Value Iteration algorithm
        for i in range(self.iterations):

            temp_counter = util.Counter(self.values)

            # Go through all states
            for state in state_list:

                # print('start state ', state)

                actions = self.mdp.getPossibleActions(state)

                # Find the maximum Q-value for the state
                max_Qvalue = float("-inf")

                # If the state is the terminal state, no more future rewards
                if self.mdp.isTerminal(state):
                    continue
                else:
                    for act in actions:
                        temp_Qvalue = self.computeQValueFromValues(state, act)

                        if temp_Qvalue > max_Qvalue:
                            max_Qvalue = temp_Qvalue

                temp_counter[state] = max_Qvalue

            self.values = temp_counter


    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        "*** YOUR CODE HERE ***"

        q_val = 0

        state_action_pairs = self.mdp.getTransitionStatesAndProbs(state, action)

        for pair in state_action_pairs:

            q_val += pair[1] * (self.mdp.getReward(state, action, pair[0]) + self.discount * self.getValue(pair[0]))

        # print('current state is ', state, ' with action ', action)
        # print('Q value is ', q_val)


        return q_val

        util.raiseNotDefined()

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        "*** YOUR CODE HERE ***"

        if self.mdp.isTerminal(state):
            return None

        actions_list = self.mdp.getPossibleActions(state)

        vals = util.Counter()
        for act in actions_list:
            vals[act] = self.getQValue(state, act)

        return vals.argMax()

        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)

class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        state_list = self.mdp.getStates()

        # Start Asynchronous Value Iteration algorithm
        # Update one state each iteration
        for i in range(self.iterations):

            idx = i % len(state_list)   # Loop back to update the rest of the states
            actions = self.mdp.getPossibleActions(state_list[idx])

            # Find the maximum Q-value for the state
            max_Qvalue = float("-inf")

            # If the state is the terminal state, no more future rewards
            if self.mdp.isTerminal(state_list[idx]):
                continue
            else:
                for act in actions:
                    temp_Qvalue = self.computeQValueFromValues(state_list[idx], act)

                    if temp_Qvalue > max_Qvalue:
                        max_Qvalue = temp_Qvalue

                self.values[state_list[idx]] = max_Qvalue




class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100, theta = 1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        "*** YOUR CODE HERE ***"

        # Create a dictionary to hold predecessors of states
        predec = {}

        states = self.mdp.getStates()

        # Compute predecessors of all states
        for st in states:
            actions = self.mdp.getPossibleActions(st)

            for act in actions:     # All possible actions at state st
                stateProb = self.mdp.getTransitionStatesAndProbs(st, act)

                for pair in stateProb:      # All possible next actions at state_action pair (st, act)
                    # If the probability is not 0
                    if pair[1] != 0:

                        # If the dictionary already contains state pair[0]
                        # Update the predec \dictionary\ content, add another predecessor state st

                        # print('pair[0] is ', pair[0], ' keys are ', predec.keys())
                        if pair[0] in predec.keys():

                            og_content = predec[pair[0]]

                            # print('og content is ', og_content)

                            predec[pair[0]] = og_content.union({st})    # og_content should be of type SET
                        else:
                            predec[pair[0]] = {st}

                        # print('we assigned predec[pair[0]] with ', predec[pair[0]])



