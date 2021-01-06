#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Sep 15 20:34:07 2020

@author: harshita
"""
"""
Search (Chapters 3-4)

The way to use this code is to subclass Problem to create a class of problems,
then create problem instances and solve them with calls to the various search
functions.
"""

import sys
from collections import deque 
from utils import *


class Problem:
    """The abstract class for a formal problem. You should subclass
    this and implement the methods actions and result, and possibly
    __init__, goal_test, and path_cost. Then you will create instances
    of your subclass and solve them with the various search functions."""

    def __init__(self, initial, goal=None):
        """The constructor specifies the initial state, and possibly a goal
        state, if there is a unique goal. Your subclass's constructor can add
        other arguments."""
        self.initial = initial
        self.goal = goal

    def actions(self, state):
        """Return the actions that can be executed in the given
        state. The result would typically be a list, but if there are
        many actions, consider yielding them one at a time in an
        iterator, rather than building them all at once."""
        child_states = list()
        
        if state[2] == 'L':  
            if(state[0] >= 2):
                if ((state[1] <= (state[0]-2)) or state[0] == 2) and (state[4] <= (state[3]+2)):
                    child_states.append([state[0] - 2, state[1], 'R', state[3]+2, state[4]])
            if(state[0] >= 1):
                if ((state[1] <= (state[0]-1)) or state[0] == 1) and (state[4] <= (state[3]+1)):
                    child_states.append([state[0] - 1, state[1],'R', state[3]+1, state[4]])
            if(state[1] >= 2):
                if ((state[3] >= (state[4] +2)) or state[3] == 0):
                    child_states.append([state[0], state[1]-2, 'R', state[3], state[4]+2])
            if(state[1] >= 1):
                if (state[3] >= (state[4] +1)) or state[3] == 0:
                    child_states.append([state[0], state[1]-1,'R', state[3], state[4]+1])
            if((state[0] >= 1 and state[1] >= 1)) and ((state[4]+1) <= (state[3]+1)):
                child_states.append([state[0]-1, state[1]-1,'R', state[3]+1, state[4]+1])
            
        else: 
            if(state[3] >= 2):
                if (state[4] <= (state[3]-2) or state[3] == 2) and (state[1] <= (state[0]+2)):
                    child_states.append([state[0] + 2, state[1],'L', state[3]-2, state[4]])
            if(state[3] >= 1):
                if (state[4] <= (state[3]-1) or state[3] == 1) and (state[1] <= (state[0]+1)):
                    child_states.append([state[0] + 1, state[1],'L', state[3]-1, state[4]])
            if(state[4] >= 2):
                if state[0] >= (state[1] +2) or state[0] == 0:
                    child_states.append([state[0], state[1]+2,'L', state[3], state[4]-2 ])
            if(state[4] >= 1):
                if state[0] >= (state[1] +1) or state[0] == 0:
                    child_states.append([state[0], state[1]+1, 'L', state[3], state[4]-1])
            if(state[3] >= 1 and state[4] >= 1) and ((state[0]+1) <= (state[1]+1)):
                child_states.append([state[0]+1, state[1]+1, 'L', state[3]-1, state[4]-1])
            
        return child_states

    def result(self, state, action):
        """Return the state that results from executing the given
        action in the given state. The action must be one of
        self.actions(state)."""
        return action

    def goal_test(self, state):
        """Return True if the state is a goal. The default method compares the
        state to self.goal or checks for state in self.goal if it is a
        list, as specified in the constructor. Override this method if
        checking against a single self.goal is not enough."""
        return state == self.goal

    def path_cost(self, c, state1, action, state2):
        """Return the cost of a solution path that arrives at state2 from
        state1 via action, assuming cost c to get up to state1. If the problem
        is such that the path doesn't matter, this function will only look at
        state2. If the path does matter, it will consider c and maybe state1
        and action. The default method costs 1 for every step in the path."""
        return c + 1

    def value(self, state):
        """For optimization problems, each state has a value. Hill Climbing
        and related algorithms try to maximize this value."""
        raise NotImplementedError


# ______________________________________________________________________________


class Node:
    """A node in a search tree. Contains a pointer to the parent (the node
    that this is a successor of) and to the actual state for this node. Note
    that if a state is arrived at by two paths, then there are two nodes with
    the same state. Also includes the action that got us to this state, and
    the total path_cost (also known as g) to reach the node. Other functions
    may add an f and h value; see best_first_graph_search and astar_search for
    an explanation of how the f and h values are handled. You will not need to
    subclass this class."""

    def __init__(self, state, parent=None, action=None, path_cost=0):
        """Create a search tree Node, derived from a parent by an action."""
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost
        self.depth = 0
        if parent:
            self.depth = parent.depth + 1

    def __repr__(self):
        return "<Node {}>".format(self.state)

    def __lt__(self, node):
        return self.state < node.state

    def expand(self, problem):
        """List the nodes reachable in one step from this node."""
        return [self.child_node(problem, action)
                for action in problem.actions(self.state)]

    def child_node(self, problem, action):
        """[Figure 3.10]"""
        next_state = problem.result(self.state, action)
        next_node = Node(next_state, self, action, problem.path_cost(self.path_cost, self.state, action, next_state))
        return next_node

    def solution(self):
        """Return the sequence of actions to go from the root to this node."""
        return [node.action for node in self.path()[1:]]

    def path(self):
        """Return a list of nodes forming the path from the root to this node."""
        node, path_back = self, []
        while node:
            path_back.append(node)
            node = node.parent
        return list(reversed(path_back))

    # We want for a queue of nodes in breadth_first_graph_search or
    # astar_search to have no duplicated states, so we treat nodes
    # with the same state as equal. [Problem: this may not be what you
    # want in other contexts.]

    def __eq__(self, other):
        return isinstance(other, Node) and self.state == other.state

    def __hash__(self):
        # We use the hash value of the state
        # stored in the node instead of the node
        # object itself to quickly search a node
        # with the same state in a Hash Table
        return hash(self.state)


def Path_Route_Result(goal):
    print("\nGoal State Reached!")
    print("Path to the goal from the initial state")
    path = list()
    node = goal
    while node.parent != None:
        path.append(node)
        node = node.parent
    path.append(node)
    while path:
        print(path.pop().state)
    print("\n")
    
# ______________________________________________________________________________
# Uninformed Search algorithms
    
def breadth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the shallowest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = deque([Node(problem.initial)])  # FIFO queue
    while frontier:
        node = frontier.popleft()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def depth_first_tree_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Repeats infinitely in case of loops.
    """

    frontier = [Node(problem.initial)]  # Stack

    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        frontier.extend(node.expand(problem))
    return None

def depth_first_graph_search(problem):
    """
    [Figure 3.7]
    Search the deepest nodes in the search tree first.
    Search through the successors of a problem to find a goal.
    The argument frontier should be an empty queue.
    Does not get trapped by loops.
    If two paths reach a state, only use the first one.
    """
    frontier = [(Node(problem.initial))]  # Stack

    explored = set()
    while frontier:
        node = frontier.pop()
        if problem.goal_test(node.state):
            return node
        explored.add(node.state)
        frontier.extend(child for child in node.expand(problem)
                        if child.state not in explored and child not in frontier)
    return None
    
def breadth_first_graph_search(problem):
    """[Figure 3.11]
    Note that this function can be implemented in a
    single line as below:
    return graph_search(problem, FIFOQueue())
    """
    node = Node(problem.initial)
    if problem.goal_test(node.state):
        return node
    frontier = deque([node])
    explored = set()
    while frontier:
        node = frontier.popleft()
        explored.add(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                if problem.goal_test(child.state):
                    return child
                frontier.append(child)
    return None    

def best_first_graph_search(problem, f, display=False):
    """Search the nodes with the lowest f scores first.
    You specify the function f(node) that you want to minimize; for example,
    if f is a heuristic estimate to the goal, then we have greedy best
    first search; if f is node.depth then we have breadth-first search.
    There is a subtlety: the line "f = memoize(f, 'f')" means that the f
    values will be cached on the nodes as they are computed. So after doing
    a best first search you can examine the f values of the path returned."""
    f = memoize(f, 'f')
    node = Node(problem.initial)
    frontier = PriorityQueue('min', f)
    frontier.append(node)
    explored = list()
    frontier_node = list()
    frontier_node.append(node.state)
    step = 0
    while frontier:
        node = frontier.pop()
        if step < 5:
            print("\nStep", step+1)
            print("Current Node: ", node.state)
            print("Children of the Current Node:")
            for child in node.expand(problem):
                    print(child.state)
        step += 1
        frontier_node.remove(node.state)
        if problem.goal_test(node.state):
            Path_Route_Result(node)
            if display:
                print(len(explored), "paths have been expanded and", len(frontier), "paths remain in the frontier")
            return node
        explored.append(node.state)
        for child in node.expand(problem):
            if child.state not in explored and child not in frontier:
                frontier.append(child)
                frontier_node.append(child.state)
            elif child in frontier:
                if f(child) < frontier[child]:
                    frontier_node.remove(frontier[child].state)
                    del frontier[child]
                    frontier.append(child)
                    frontier_node.append(child.state)
        if step <= 5:
            print("Frontier Content :", frontier_node)
            print("Explored Content :",explored)
    return None


def uniform_cost_search(problem, display=False):
    """[Figure 3.14]"""
    return best_first_graph_search(problem, lambda node: node.path_cost, display)


def depth_limited_search(problem, limit=50):
    """[Figure 3.17]"""
    
    def recursive_dls(node, problem, limit):
        if problem.goal_test(node.state):
            Path_Route_Result(node)
            return node
        elif limit == 0:
            return 'cutoff'
        else:
            cutoff_occurred = False
            for child in node.expand(problem):
                result = recursive_dls(child, problem, limit - 1)
                if result == 'cutoff':
                    cutoff_occurred = True
                elif result is not None:
                    return result
            return 'cutoff' if cutoff_occurred else None
        

    # Body of depth_limited_search:
    return recursive_dls(Node(problem.initial), problem, limit)


def iterative_deepening_search(problem):
    """[Figure 3.18]"""
    for depth in range(sys.maxsize):
        result = depth_limited_search(problem, depth)
        if result != 'cutoff':
            return result
# ______________________________________________________________________________
# Informed (Heuristic) Search

greedy_best_first_graph_search = best_first_graph_search

# Greedy best-first search is accomplished by specifying f(n) = h(n).

def astar_search(problem, h=None, display=False):
    """A* search is best-first graph search with f(n) = g(n)+h(n).
    You need to specify the h function when you call astar_search, or
    else in your Problem subclass."""
    h = memoize(h or problem.h, 'h')
    return best_first_graph_search(problem, lambda n: n.path_cost + h(n), display)

# ______________________________________________________________________________
# Other search algorithms
    
def recursive_best_first_search(problem, h=None):
    """[Figure 3.26]"""
    h = memoize(h or problem.h, 'h') 

    def RBFS(problem, node, flimit, explored, f):
        if problem.goal_test(node.state):
            Path_Route_Result(node)
            return node, 0, explored, f  # (The second value is immaterial)
        
        successors = node.expand(problem)
        if len(successors) == 0:
            return None, np.inf, explored, f       
        for s in successors:
            s.f = max(s.path_cost + h(s), node.f)
            
        if len(explored) <= 5:
                print("\nStep", len(explored))
                print("Current Node: ", node.state)
                print("Children of Current Node: ")
                for s in successors:
                    f.append(s.state)
                    print(s.state)
                temp = []
                dup = explored[-1]
                if type(dup[0]) != list:
                    temp.append(dup)
                else:
                    for l in dup:
                        temp.append(l)
                    temp.append(node.state)
                explored.extend([temp])
                print("Explored Content:",explored[-1])
                for i in explored[-1]:
                    f[:] = (val for val in f if val != i)
                temp = []
                for frontier in f:
                    frontier = tuple(frontier)
                    temp.append(frontier)
                print("Frontier Content:",set(temp))
                
        while True:
            # Order by lowest f value
            successors.sort(key=lambda x: x.f)
            alternative = 0
            if len(successors) > 1:
                alternative = successors[1].f
            else:
                alternative = np.inf
            best = successors[0]
            if best.f > flimit:
                return None, best.f, explored, f
            result, best.f, expl, f = RBFS(problem, best, min(flimit, alternative), explored, f)
            if result is not None:
                return result, best.f, expl, f

    node = Node(problem.initial)
    node.f = h(node)
    result, bestf, expl, f = RBFS(problem, node, np.inf, [node.state], [])
    return result

def heuristic(n):
        return (n.state[0]+n.state[1]-1)

def main():
    if (len(sys.argv)) != 3:
        print("Invalid number of arguments.")
        sys.exit()
    missionaries = int(sys.argv[1].strip())
    cannibals = int(sys.argv[2].strip())
    if missionaries < cannibals:
        print("Initial state has more cannibals than missionaries.")
        sys.exit()
    
    initial = [missionaries,cannibals,'L',0,0]
    goal = [0,0,'R',missionaries,cannibals]
    
    print("Uniform Cost Search:")
    uniform_cost_search(Problem(initial, goal))
    
    print("Iterative Deepening Search:")
    iterative_deepening_search(Problem(initial, goal))
    
    print("Greedy Best-First Search: ")
    greedy_best_first_graph_search(Problem(initial, goal), heuristic)
    
    print("A* Search:")
    astar_search(Problem(initial, goal), heuristic)
    
    print("Recursive Best-First Search:")
    recursive_best_first_search(Problem(initial, goal), heuristic)

main()