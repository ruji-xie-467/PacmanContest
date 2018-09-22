# myTeam.py
# ---------
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


from captureAgents import CaptureAgent
import random, time, util
from game import Directions
import game
from game import Agent
import math
from util import nearestPoint

#################
# Team creation #
#################

def createTeam(firstIndex, secondIndex, isRed,
               first = 'AlphaBetaAgent', second = 'DefensiveReflexAgent'):
  """
  This function should return a list of two agents that will form the
  team, initialized using firstIndex and secondIndex as their agent
  index numbers.  isRed is True if the red team is being created, and
  will be False if the blue team is being created.

  As a potentially helpful development aid, this function can take
  additional string-valued keyword arguments ("first" and "second" are
  such arguments in the case of this function), which will come from
  the --redOpts and --blueOpts command-line arguments to capture.py.
  For the nightly contest, however, your team will be created without
  any extra arguments, so you should make sure that the default
  behavior is what you want for the nightly contest.
  """
  # The following line is an example only; feel free to change it.
  return [eval(first)(firstIndex), eval(second)(secondIndex)]

##########
# Agents #
##########

class DummyAgent(CaptureAgent):
  """
  A Dummy agent to serve as an example of the necessary agent structure.
  You should look at baselineTeam.py for more details about how to
  create an agent as this is the bare minimum.
  """

  def registerInitialState(self, gameState):
    """
    This method handles the initial setup of the
    agent to populate useful fields (such as what team
    we're on).

    A distanceCalculator instance caches the maze distances
    between each pair of positions, so your agents can use:
    self.distancer.getDistance(p1, p2)

    IMPORTANT: This method may run for at most 15 seconds.
    """

    '''
    Make sure you do not delete the following line. If you would like to
    use Manhattan distances instead of maze distances in order to save
    on initialization time, please take a look at
    CaptureAgent.registerInitialState in captureAgents.py.
    '''
    #CaptureAgent.registerInitialState(self, gameState)

    '''
    Your initialization code goes here, if you need any.
    '''
    self.start = gameState.getAgentPosition(self.index)
    CaptureAgent.registerInitialState(self, gameState)


  def chooseAction(self, gameState):
    """
    Picks among actions randomly.
    """
    actions = gameState.getLegalActions(self.index)

    '''
    You should change this in your own agent.
    '''
    actions = gameState.getLegalActions(self.index)

    # You can profile your evaluation time by uncommenting these lines
    # start = time.time()
    values = [self.evaluate(gameState, a) for a in actions]
    # print 'eval time for agent %d: %.4f' % (self.index, time.time() - start)

    maxValue = max(values)
    bestActions = [a for a, v in zip(actions, values) if v == maxValue]

    foodLeft = len(self.getFood(gameState).asList())

    if foodLeft <= 2:
      bestDist = 9999
      for action in actions:
        successor = self.getSuccessor(gameState, action)
        pos2 = successor.getAgentPosition(self.index)
        dist = self.getMazeDistance(self.start, pos2)
        if dist < bestDist:
          bestAction = action
          bestDist = dist
      return bestAction

    return random.choice(bestActions)

    #return random.choice(actions)



  def getSuccessor(self, gameState, action):
    """
    Finds the next successor which is a grid position (location tuple).
    """
    successor = gameState.generateSuccessor(self.index, action)
    pos = successor.getAgentState(self.index).getPosition()
    if pos != nearestPoint(pos):
      # Only half a grid position was covered
      return successor.generateSuccessor(self.index, action)
    else:
      return successor

  def evaluate(self, gameState, action):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState, action)
    weights = self.getWeights(gameState, action)

    # previousObservation = self.getPreviousObservation()
    # if previousObservation:
    #   if previousObservation.getAgentPosition(self.index) == gameState.getAgentPosition(self.index):
    #     print("I am Agnet:", self.index, "I am at: ", gameState.getAgentPosition(self.index), " score:", features * weights)
    #     print(features)
    #     print('action', action)
    if gameState.isRed(gameState.getAgentPosition(self.index)) and not gameState.isOnRedTeam(self.index):
      print("I am Agnet:", self.index, "I am at: ", gameState.getAgentPosition(self.index), " score:", features * weights)
      print(features)
      print('action', action)
    return features * weights

  def getFeatures(self, gameState, action):
    """
    Returns a counter of features for the state
    """
    features = util.Counter()

    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    features['successorScore'] = successor
    return features

  def getWeights(self, gameState, action):
    """
    Normally, weights do not depend on the gamestate.  They can be either
    a counter or a dictionary.
    """
    return {'successorScore': 1.0}

class MultiAgentSearchAgent(CaptureAgent):
  """
  This class provides some common elements to all of your
  multi-agent searchers.  Any methods defined here will be available
  to the MinimaxPacmanAgent, AlphaBetaPacmanAgent & ExpectimaxPacmanAgent.

  You *do not* need to make any changes here, but you can if you want to
  add functionality to all your adversarial search agents.  Please do not
  remove anything, however.

  Note: this is an abstract class: one that should not be instantiated.  It's
  only partially specified, and designed to be extended.  Agent (game.py)
  is another abstract class.
  """

  def evaluate(self, gameState):
    """
    Computes a linear combination of features and feature weights
    """
    features = self.getFeatures(gameState)
    weights = self.getWeights(gameState)

    # previousObservation = self.getPreviousObservation()
    # if previousObservation:
    #   if previousObservation.getAgentPosition(self.index) == gameState.getAgentPosition(self.index):
    #     print("I am Agnet:", self.index, "I am at: ", gameState.getAgentPosition(self.index), " score:", features * weights)
    #     print(features)
    #     print('action', action)
    #print("I am Agnet:", self.index, "I am at: ", gameState.getAgentPosition(self.index), " score:", features * weights)
    #print(features)
    # print('action', action)
    # if gameState.isRed(gameState.getAgentPosition(self.index)) and not gameState.isOnRedTeam(self.index):
    #   print("I am Agnet:", self.index, "I am at: ", gameState.getAgentPosition(self.index), " score:", features * weights)
      # print(features)
    # if not features * weights:
    #   return -math.inf
    return features * weights

class AlphaBetaAgent(MultiAgentSearchAgent):
  """
  Your minimax agent with alpha-beta pruning (question 3)
  """
  def minimaxTree_node(self, gameState, k, maxDepth, parrentNode, alpha, beta):
    n = gameState.getNumAgents()
    depth = k // n + 1

    if gameState.isOver() or depth > maxDepth and k % n == self.index:
      # if gameState.isWin():
      #     print("win?? you must be kidding")
      #     print("ghost position: ", [gameState.getGhostPosition(i) for i in range(1,4)])
      #     print("pacman position: ", gameState.getPacmanPosition())
      # elif gameState.isLose():
      #     print("I am dead, of course")
      #     print("ghost position: ", [gameState.getGhostPosition(i) for i in range(1,4)])
      #     print("pacman position: ", gameState.getPacmanPosition())
      # else:
      #     print("cannot dive")
      #print(gameState.getAgentPosition(self.index))
      return self.evaluate(gameState)

    agentIndex = k % n
    actionList = gameState.getLegalActions(agentIndex)

    if agentIndex == self.index:  # pacman
      maxscore = -math.inf
      for action in actionList:
        nextState = gameState.generateSuccessor(agentIndex, action)
        # print("pacman action: ", action, "pacman position: ", gameState.getPacmanPosition())
        thisActionTreeNode = [[action], []]
        score = self.minimaxTree_node(nextState, k + 1, maxDepth, thisActionTreeNode, alpha, beta)
        maxscore = max(score, maxscore)
        if maxscore > beta:
          return maxscore
        alpha = max(alpha, maxscore)
        # print("I am at:", gameState.getAgentPosition(1), "action", action)
        thisActionTreeNode[0].append(score)  # [[action, score], []]
        parrentNode[1].append(thisActionTreeNode)
      return maxscore
    elif gameState.isRed(gameState.getAgentPosition(agentIndex)) and gameState.isOnRedTeam(agentIndex):  # ghost
      minscore = math.inf
      for action in actionList:
        nextState = gameState.generateSuccessor(agentIndex, action)
        # print("ghost index: ", agentIndex, "ghost position: ", gameState.getGhostPosition(agentIndex))
        thisActionTreeNode = [[action], []]
        score = self.minimaxTree_node(nextState, k + 1, maxDepth, thisActionTreeNode, alpha, beta)
        minscore = min(minscore, score)
        if minscore < alpha:
          return minscore
        beta = min(beta, minscore)
        thisActionTreeNode[0].append(score)  # [[action, score], []]
        parrentNode[1].append(thisActionTreeNode)
      return minscore
    else:
      return self.minimaxTree_node(gameState, k + 1, maxDepth, parrentNode, alpha, beta)

  def findPacmanPath(self, gameState, treeNode, maxDepth, k, actions):
    n = gameState.getNumAgents()
    goDeep = k // n
    agentIndex = k % n
    if goDeep > maxDepth:
      return
    if not treeNode[1]: return
    if agentIndex == self.index:
      maxScore = - math.inf
      for i in range(len(treeNode[1])):
        if treeNode[1][i][0][1] > maxScore:
          maxScore = treeNode[1][i][0][1]  # [1]: child node list, [i]: ith child node, [0]: child node action and score, [1]: child node score
          action = treeNode[1][i][0][0]
          index = i
      actions.append(action)
      self.findPacmanPath(gameState, treeNode[1][index], maxDepth, k + 1, actions)
    elif gameState.isRed(gameState.getAgentPosition(agentIndex)) and gameState.isOnRedTeam(agentIndex):
      minScore = math.inf
      for i in range(len(treeNode[1])):
        if treeNode[1][i][0][1] < minScore:
          minScore = treeNode[1][i][0][1]
          index = i
      self.findPacmanPath(gameState, treeNode[1][index], maxDepth, k + 1, actions)
    else:
      self.findPacmanPath(gameState, treeNode, maxDepth, k + 1, actions)

  def getAction(self, gameState):
    """
    Returns the minimax action using self.depth and self.evaluationFunction
    """
    "*** YOUR CODE HERE ***"
    maxDepth = 2
    myAgentIndex = gameState.getBlueTeamIndices()
    n = gameState.getNumAgents()
    tree = [["first"], []]
    finalscore = self.minimaxTree_node(gameState, 0, maxDepth, tree, -math.inf, math.inf)
    # print("finalscore: ", finalscore)
    # print(maxDepth)
    actions = []
    self.findPacmanPath(gameState, tree, maxDepth, 0, actions)
    # print("action: ", actions)
    return actions[0]

  def getWeights(self, gameState):
    return {'successorScore': 20, 'distanceToFood': -10, 'rDistanceToGhost': -50}

  def getFeatures(self, gameState):
    features = util.Counter()
    foodList = self.getFood(gameState).asList()
    features['successorScore'] = -len(foodList)  # self.getScore(successor)
    if len(foodList) > 0:  # This should always be True,  but better safe than sorry
      myPos = gameState.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    newPos = gameState.getAgentPosition(self.index)
    newFood = gameState.getBlueFood()
    redIndex = gameState.getRedTeamIndices()
    blueIndex = gameState.getBlueTeamIndices()
    newGhostStates = []
    for index in redIndex:
      ghost = gameState.getAgentState(index)
      if gameState.isRed(ghost.getPosition()):
        # print(ghost)
        newGhostStates.append(gameState.getAgentState(index))
      newPellet = gameState.getCapsules()
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    minGhostDistance = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in newGhostStates])

    features['rDistanceToGhost'] = 1 / minGhostDistance

    # if len(gameState.getLegalActions(self.index)) == 2:
    #   features['inPit'] = 1
      # features['successorScore'] = self.getScore(successor)

    return features

class OffensiveReflexAgent(DummyAgent):
  """
  A reflex agent that seeks food. This is an agent
  we give you to get an idea of what an offensive agent might look like,
  but it is by no means the best or only way to build an offensive agent.
  """
  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)
    foodList = self.getFood(successor).asList()
    features['successorScore'] = -len(foodList)#self.getScore(successor)
    if len(foodList) > 0: # This should always be True,  but better safe than sorry
      myPos = successor.getAgentState(self.index).getPosition()
      minDistance = min([self.getMazeDistance(myPos, food) for food in foodList])
      features['distanceToFood'] = minDistance

    newPos = successor.getAgentPosition(self.index)
    newFood = successor.getBlueFood()
    redIndex = successor.getRedTeamIndices()
    blueIndex = successor.getBlueTeamIndices()
    newGhostStates = []
    for index in redIndex:
      ghost = successor.getAgentState(index)
      if successor.isRed(ghost.getPosition()):
        #print(ghost)
        newGhostStates.append(successor.getAgentState(index))
      newPellet = successor.getCapsules()
      newScaredTimes = [ghostState.scaredTimer for ghostState in newGhostStates]
    minGhostDistance = min([self.getMazeDistance(myPos, ghost.getPosition()) for ghost in newGhostStates])

    features['rDistanceToGhost'] = 1/minGhostDistance

    if len(successor.getLegalActions(self.index)) == 2:
      features['inPit'] = 1
      #features['successorScore'] = self.getScore(successor)

    return features

  def getWeights(self, gameState, action):
    return {'successorScore': 100, 'distanceToFood': -1, 'rDistanceToGhost': -100, 'inPit': -10000}

class DefensiveReflexAgent(DummyAgent):
  """
  A reflex agent that keeps its side Pacman-free. Again,
  this is to give you an idea of what a defensive agent
  could be like.  It is not the best or only way to make
  such an agent.
  """

  def getFeatures(self, gameState, action):
    features = util.Counter()
    successor = self.getSuccessor(gameState, action)

    myState = successor.getAgentState(self.index)
    myPos = myState.getPosition()

    # Computes whether we're on defense (1) or offense (0)
    features['onDefense'] = 1
    if myState.isPacman: features['onDefense'] = 0

    # Computes distance to invaders we can see
    enemies = [successor.getAgentState(i) for i in self.getOpponents(successor)]
    invaders = [a for a in enemies if a.isPacman and a.getPosition() != None]
    features['numInvaders'] = len(invaders)
    if len(invaders) > 0:
      dists = [self.getMazeDistance(myPos, a.getPosition()) for a in invaders]
      features['invaderDistance'] = min(dists)

    if action == Directions.STOP: features['stop'] = 1
    rev = Directions.REVERSE[gameState.getAgentState(self.index).configuration.direction]
    if action == rev: features['reverse'] = 1

    return features

  def getWeights(self, gameState, action):
    return {'numInvaders': -1000, 'onDefense': 100, 'invaderDistance': -10, 'stop': -100, 'reverse': -2}


def manhattanDistance(xy1, xy2):
  "The Manhattan distance heuristic for a PositionSearchProblem"
  return abs(xy1[0] - xy2[0]) + abs(xy1[1] - xy2[1])
