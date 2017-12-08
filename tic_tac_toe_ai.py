#!/usr/bin/env python
import random
import math
import gameworks
import numpy as np
import pickle
import sys


# 1 = player 1
# 2 = player 2
# 0 = no move yet

Winning_boards_init = [
  
  # player 0 wins
  
  [
   -1, -1, -1, # p0 wins by row one
   0, 0, 0,
   0, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [
   0, 0, 0, 
   -1, -1, -1,
   0, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [
   0, 0, 0, 
   0, 0, 0,
   -1, -1, -1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [
   -1, 0, 0, 
   -1, 0, 0,
   -1, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [ 
   0, -1, 0, 
   0, -1, 0,
   0, -1, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [
   0, 0, -1, 
   0, 0, -1,
   0, 0, -1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [ 
   -1, 0, 0, 
   0, -1, 0,
   0, 0, -1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [ 
   0, 0, -1, 
   0, -1, 0,
   -1, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
# Player 1 wins

  [
   1, 1, 1, # p1 wins by row one
   0, 0, 0,
   0, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [ 
   0, 0, 0, 
   1, 1, 1,
   0, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [
   0, 0, 0, 
   0, 0, 0,
   1, 1, 1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [
   1, 0, 0, 
   1, 0, 0,
   1, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [ 
   0, 1, 0, 
   0, 1, 0,
   0, 1, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [
   0, 0, 1, 
   0, 0, 1,
   0, 0, 1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],
  
  [ 
   1, 0, 0, 
   0, 1, 0,
   0, 0, 1,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  [
   0, 0, 1, 
   0, 1, 0,
   1, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
   ],

  
  
  # mixed wins
  
  [
    1, -1,  1,
   -1, -1,  1,
   -1,  0,  1,
   0
   ],
  
  [
   -1,  0,  1,
   -1,  1,  0,
   -1,  1, -1,
   0
  ],
  
  [
   -1,  1, -1,
    0,  1, -1,
    1,  0, -1,
    0
  ],
  
  [
    -1,  1, -1,
     0,  1,  0,
    -1,  1,  0,
   0
   ],
 
  ]

Winning_players_init_playerB = [
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   0,
   1,
   1,
   1,
   1,
   1,
   1,
   1,
   1,
   
   1,
   0,
   0,
   1,
   0
]

## debug - reset the above to nothing

Winning_boards_init = [ # this is initial state of the board... not a winning board!
   ]

Winning_players_init_playerB = [
]


Losing_boards_init = [ # these are games that are losing plays for both players
  [
   0, 0, 0, 
   0, 0, 0,
   0, 0, 0,
   0 # who goes next? 1=first player to move, 2=second player to move, 0=nobody..game draw or done via win/loss
  ],
  ]

Losing_players_init = [
  0
]

# perform winning prediction based on AI training in syn0 and syn1

# sigmoid function
def nonlin(x,deriv=False): # default to False, unless argument is specified 
  
    if(deriv==True):
        #print "nonlin: deriv == True"
        return x*(1-x)
      
      
    #print "nonlin: deriv == false"
    
    return 1/(1+np.exp(-x))
  
  
# DO TRAINING

def do_train(X, y, thisPlayer, training_batch_count):
  
  training_sample_count = len(X)
  node_count = len(X[0])
  
  print "X has ", training_sample_count, " training samples, and ", node_count, " nodes"
  
  Debug = False
      
  
  if (len(y)!= training_sample_count):
    print "ERROR - number of training samples=", training_sample_count, " does not match output dataset = ", len(y)
    exit
  
  # seed random numbers to make calculation
  # deterministic (just a good practice)
  np.random.seed(1)
  
  if (training_batch_count > training_sample_count):
    training_batch_count =  training_sample_count # training_sample_count # small batch we will process below (1 by 1) and accum into thisPlayer.syn[] lists
    
  accum_syn = [] # list to accum weights for syn 0 and 1 per loop below
  accum_syn = [np.array([]) for i in range(2)]
  
  accum_syn[0] = np.empty([node_count,0]) # must be same dims as syn arrays in loop below, so we can concat at bottom of each loop cycle
  accum_syn[1] = np.empty([0,1])

  #print "accum_syn[0] = ", accum_syn[0]
  #print "accum_syn[1] = ", accum_syn[1]
  
  #accum_syn.append(np.zeros((node_count, training_batch_count))) # initialize a base accumulator matrix of the proper size for syn[0] weights
  #accum_syn.append(np.zeros((training_batch_count, 1))) # per above for syn[1] weights
  
  # process only one batch of training samples at a time... helps us deal with huge training sets, albeit more slowly
  
  for i in xrange(0, len(X), training_batch_count):

    xBatch = X[i:i+training_batch_count] # xBatchs are a list of training_batch_count samples for this loop cycle
    yBatch = y[i:i+training_batch_count]

    
    print "batch loop = ", i, " to  = ", (i+len(xBatch)), "(", training_batch_count, ") and xBatch = \n", xBatch
    
    syn = [] # init new list for this Player (computer first, then accum into the Player object below)
    
    # initialize weights randomly with mean 0 in -1 to 1 range.
    # We do this by first picking randomly between 0-1, scale by 2, then subtract 1
    
    # thisPlayer.syn[0] is a matrix that has node_count nodes (cols) and training_sample_count samples (rows)
    # to start: this is a matrix of random weights between -1 and 1, matching the dimensions of the training input data
    syn.append(2*np.random.random((node_count,training_batch_count)) - 1) # First layer of weights, Synapse 0, connecting l0 to l1.
    
    # thisPlayer.syn[1] is a vector that has length matching sample count
    # to start: this is a vector of randome weights between -1 and 1
    syn.append(2*np.random.random((training_batch_count,1)) - 1) # Second layer of weights, Synapse 1 connecting l1 to l2.
    
    for j in xrange(60000):
      if (Debug==True):
        print "\n\n*********************"
        print "\nIteration #", j
        print "\nSynapse 0 weights:"
        print syn[0]
  
      # Feed forward through layers 0, 1, and 2
      l0 = xBatch
      l1 = nonlin(np.dot(l0,syn[0])) # matrix * matrix = matrix, then we "scale" these -1->1 weights by the sigmoid funciton
      l2 = nonlin(np.dot(l1,syn[1])) # matrix * vector = vector, here we take the "scaled" matrix of l1 weights and dot with a vector, then "rescale" by the sigmoid function to product l2
      
      # l2 (vector) is our estimated "answer" for every training sample represented in the vector y
      # we can compare l2 to y to seee how close we go...
      
      if (Debug==True):
        print "\nLayer 0:"
        print l0
        print "\nLayer 1:"
        print l1
  
      # how much did we miss?
      l2_error = yBatch - l2 # diff between output dataset and second layer (vector - vector = vector)
      
      if (Debug==True):
        print "\nl2 Error:"
        print l2_error
        
        
      if (j% 10000) == 0:
          print "Mean Err:" + str(np.mean(np.abs(l2_error))) # debug output at zero and at every 10000 iterations
  
      # in what direction is the target value?
      # were we really sure? if so, don't change too much.
      l2_delta = l2_error*nonlin(l2,deriv=True) # This is the error of the network scaled by the confidence. It's almost identical to the error except that very confident errors are muted.
      # ^ a vector
  
  
      # how much did each l1 value contribute to the l2 error (according to the weights)?
      l1_error = l2_delta.dot(syn[1].T) # Weighting l2_delta by the weights in syn[1], we can calculate the error in the middle/hidden layer.
      # ^ a scalar
      
      # in what direction is the target l1?
      # were we really sure? if so, don't change too much.
      l1_delta = l1_error * nonlin(l1,deriv=True) # This is the l1 error of the network scaled by the confidence. Again, it's almost identical to the l1_error except that confident errors are muted.
      # ^ matrix
  
  # NOTE: * (above) = Elementwise multiplication, so two vectors of equal size
  #   are multiplying corresponding values 1-to-1 to generate a final vector of identical size.
  
      # update weights
      syn[1] += l1.T.dot(l2_delta) # matrix . vector = vector
      syn[0] += l0.T.dot(l1_delta) # matrix . matrix = matrix
      
      # NOTE: for x.dot(y) If x and y are vectors, this is a dot product.
      #  If both are matrices, it's a matrix-matrix multiplication.
      # If only one is a matrix, then it's vector matrix multiplication.
      
    # end: J loop
    
    print "pre accum:"
    print " syn[0] = ", syn[0]
    print " syn[1] = ", syn[1]
    #print " accum_syn[0] = ", accum_syn[0]
    #print " accum_syn[1] = ", accum_syn[1]
    
    # now accumulate syn[0] and syn[1] into accum_syn

    accum_syn[0] = np.concatenate((accum_syn[0], syn[0]), axis=1)

    
    #print "post accum"
    #print " accum_syn[0] = ", accum_syn[0]
    
    #for q in range(0,len(syn[0])):
    #  accum_syn[0][q] += syn[0][q] # concat new batch
      
    #for q in syn[1]:
    #  accum_syn[1] = np.append(accum_syn[1], q)

    accum_syn[1] = np.concatenate((accum_syn[1], syn[1]), axis=0)
    
    #print " accum_syn[1] = ", accum_syn[1]
  
  
  # end: xBatch small batch loop
  
  print "\nTraining Input for player ", thisPlayer
  print X
  print "\nTraining Output for player ", thisPlayer
  print y
  print "\n\nOutput After Training for player ", thisPlayer
  print "\nL1 = "
  print l1
  print "\nL2 = "
  print l2
  print "\n\Synaps After Training for player ", thisPlayer
  
  #print ("\nsyn0_player%s = np.array( [" % pnum)
  #for i in syn[0]:
  #  print "["
  #  for j in i:
  #    print j, ", ",
  #  print "],"
  #print "])"
      

  #print ("\nsyn1_player%s = np.array( [" % pnum)
  #for i in syn[1]:
  #  print i, ", "
  #print "])"
  
  thisPlayer.syn = accum_syn

  print "**thisPlayer.syn[0] = "
  print thisPlayer.syn[0]
  
  print "**thisPlayer.syn[1] ="
  print thisPlayer.syn[1]
  
  #
  #new_game = np.array([0, 1, -1, -1, -1, -1, -1, -1, -1, -1])
  #new_game = np.array([0, 7, 9, 1, 6, -1, -1, -1, -1, -1]) # 
  #new_game = np.array([0, 7, 4, 1, 8, 9, 5, 6, 2, -1]) # 1 wins in 8 moves
  
  # verify that training samples produce the correct result....
  
  print "**************************************"
  print "* VERIFYING RESULTS OF TRAINING ******"
  print "**************************************"
  
  warning_count = 0
  
  for i in range(0,len(X)):  
    this_test = X[i]
    l1 = nonlin(np.dot(this_test,thisPlayer.syn[0])) # vector * matrix = vector
    # l1 will be a vector with a number of scalar probability elements (0-1) that match the number of training samples used to generate syn[0]
    # syn[1] will be a matching vector shape with probabilities from (0-1)
    l2 = nonlin(np.dot(l1,thisPlayer.syn[1])) # vector . vector = scalar
  
    if (np.abs(y[i]-l2) > .01): # seems like an error..warn the user
      warning_count += 1
      print "\nWARNING #", warning_count, "(do_train)!  For player = ", thisPlayer, " For data set ", i ," with board = "
      print this_test
      print "... l1 is: \n", l1
      print "... syn[1] is: ", thisPlayer.syn[1]
      print "... prediction is: ", l2
      print "....answer should be: ", y[i]
      test_accum = 0
      for j in range(0,len(l1)):
        test_accum += l1[j]*thisPlayer.syn[1][j] #
      print ".... test accum = ", test_accum
      print ".... nonlin of test accum = ", nonlin(test_accum)
      raw_input('ok?')
    else:
      print "Training sample", i, "is ok with error =", y[i]-l2
      
      
  # store these weights in a file for future use
  
  psyn_file_name = str(thisPlayer) + '.syn.txt' # thisPlayer resolves to the string token name for this player (like x or y)
  
  with open(psyn_file_name,'w') as psyn:
    pickle.dump(thisPlayer.syn, psyn) 
  
  
  
    
# PREDICT 2: determine next best move based on game board state

def predict2(this_game, doSilent):

  print "Predicting best move for player ", this_game.board.players.whosturn(), "(", this_game.board.players.turn, ")..."
  
  # reference global player syn0 and syn1 weights for prediction, below:
  
  #syn0 = [syn_playerA[0], syn_playerB[0]]
  #syn1 = [syn_playerA[1], syn_playerB[1]]
  
  if (not doSilent):
    print "empty: ",
    for i in this_game.board.empty:
      print i, " ",
    print ""
  
  best_move_so_far = -1
  max_l2_so_far = 0.0
  player_number_chosen = -1
  
  #for each player in the game: #player id 0 or 1 (0 or A always goes first)
  
  for pid in range(0, len(this_game.board.players.players)):
    #print "Player [", pid, "].syn[0] = \n", this_game.board.players.players[pid].syn[0]
    #print "Player [", pid, "].syn[1] = \n", this_game.board.players.players[pid].syn[1]
    
      
    #print "predict2() this_game.board.players.players[pid].syn[0] = ", this_game.board.players.players[pid].syn[0]
    #print "predict2() this_game.board.players.players[pid].syn[1] = ", this_game.board.players.players[pid].syn[1]
      
    for c in this_game.board.empty: # for each unfilled space on the board for the current player
      #print "predict2(): estimating value of choosing position = ", c
      
      temp_board = []
      
      for i in this_game.board.as_array(): # copy the current game board to a temp array
        temp_board.append(i)
        
      #print "predict2() appending next player to end of temp_board: ", pid
        
      temp_board.append(0) # append id of player who will go next as the last number in the array
        
      # note we normalize board positions for the AI to be -1 (p0), 0 (no move yet), or +1 (p0)
      
      #print "predict2() setting position ", c, " to player id = ", (pid*2-1)
        
      temp_board[c-1]=pid*2-1 # set the next potentional move p from empty list to the current player id, to see what its outcome would look like
   
      #print "predict2() ready with prospective board = ",  temp_board
      
      
      l1 = nonlin(np.dot(temp_board,this_game.board.players.players[pid].syn[0])) # vector * matrix = vector
      
      #print "predict2() l1 = \n", l1
      
      dp_result = np.dot(l1,this_game.board.players.players[pid].syn[1])
      
      #print "predict2() dp_result for l2 before nonlin() = \n", dp_result
      
      l2 = nonlin(dp_result) # vector . vector = scalar
      
      #print "predict2() l2 = \n", l2
      

      if (not doSilent):
        print "For player %d (%s) the winning probability for position %s = %f vs %f with board = %s" % (pid, this_game.board.players.players[pid], this_game.board.get_row_col(c), l2, max_l2_so_far, temp_board)
      
      if (l2 > max_l2_so_far):
        best_move_so_far = c
        max_l2_so_far = l2
        player_number_chosen = pid
      
  if (not doSilent):
      print "I suggest you choose position = ", best_move_so_far, this_game.board.get_row_col(best_move_so_far), "based on player ", player_number_chosen, " probabilities"

  return best_move_so_far
    

# a game board represents the entire game - e.g. 9 characters for a 3x3 game in row order (row1, row2, row3),
#   where each cell either either empty (denoted by a special character),
#   or assigned one of the actively player's 'player' objects to indicate that that player has claimed that spot
#

class tic_tac_toe_Board(gameworks.Board ):  # the game board
    
    # initialize the tic tac toe game board
    # derive the from the general game board class in the gameworks module
    # extend the intialization method and add new methods that are specific to tic tac toe
    # initialize the board and associated sets needed to implement automated move selection and rapid checking of available spaces
    
    def __init__(self, dimension, tuple_of_players):
        super(tic_tac_toe_Board, self).__init__(dimension, tuple_of_players) # we are extending the base __init_ function from the gameworks.Board class

        self.rows = {} # create a dictionary of rows that we can use to manage set  membership (which players occupy which rows, or if empty)
        self.cols = {}
        self.diags = {}
        self.diags[1] = set() #empty set to start for the diag going left-to-right-top-to-bottom
        self.diags[2] = set() #empty set to start for the daig going left-to-right-bottom-to-top
        self.corners = set() #empty set to use for storing all for corners
        
        # initialize all set definitions for rows, columns, diagonals - used for checking progress of play
        
        for rnum in range(1,dimension+1):
            for cnum in range(1,dimension+1):
                position = self.get_position_num(rnum, cnum)
                
                if not self.rows.has_key(rnum): # create a set for this row if not yet created
                    self.rows[rnum] = set()
                    
                self.rows[rnum].add(position) #add the empty character to this row
                
                if not self.cols.has_key(cnum): # create a set for this column if not yet created
                    self.cols[cnum] = set()
                
                self.cols[cnum].add(position) # add the empty character to this column
                
                if (rnum == cnum): # this position on on the diagonal going left-right-down
                    self.diags[1].add(position) #add this position to diag1
                    
                if ((rnum+cnum)==(self.dim+1)): #this position is on the diagonal going left-right-up
                    self.diags[2].add(position) #add this position to diag2
                    
            
        
        #add the linear positions on the board of the respective corners
        self.corners.add(self.get_position_num(1,1)) #add upper left corner
        self.corners.add(self.get_position_num(1,self.dim)) #add upper right corner
        self.corners.add(self.get_position_num(self.dim, 1)) #add lower left corner
        self.corners.add(self.get_position_num(self.dim, self.dim)) # add lower right corner
        
        # find the center - only works "correctly" for odd numbered board dimensions. Even-numbered dimensions have 4 possible centers for the diagonals
        rnum = (self.dim+1)/2 # grab the center position, or near to it
        self.center = set()
        self.center.add(self.get_position_num(rnum, rnum))

    

    def ai_move(self, this_game): # pick a ai move for the active player... keep trying if the cell is actively filled, unless the game is over
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a AI move - This game is a DRAW" % player
            return False
        
        position = predict2(this_game, doSilent=True) # this position will be available by definition (unless we have an internal error)
                    
        if (self.set_position(position, player, player)): # set this position(removed from set) to the current player using player as the token - return True if successful
            print "Found a AI move"
            return True #indicates a successful update
          
    def smart_move(self, this_game): # pick a random move for the active player... keep trying if the cell is actively filled, unless the game is over
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a smart move - This game is a DRAW" % player
            return False
        
        elif self.defensive_move(this_game): #return True/False to indicate if move was successful... if false, no clear defensive move was available so try a new strategy
            return True
        
        elif (len(self.empty)==(self.dim*self.dim-1)): # 2nd move logic
            print "2nd move logic..."
            if (len(self.corners.intersection(self.empty))==3): # if openning move was a corner, grab the center
                if (self.center_move(this_game)): # try a center move
                    print "   Grab a center"
                    return True
        
            elif (len(self.center.intersection(self.empty))==0): # if openning move was a center, grab the corner
                if (self.corner_move(this_game)):
                    print "   Grab a corner"
                    return True

            else:
                print "   Random Move"
                return self.random_move(this_game) # try for any random move (will return True unless the game is a draw because all cells are occupied

                
        elif (len(self.empty) == self.dim*self.dim): # 1st move logic - if first move grab corner
                if (self.corner_move(this_game)):
                    print "1st mover logic...grab a corner"
                    return True
                
        elif (self.corner_move(this_game)): # try for a corner regardless of # moves, if available
            print "Nth move logic.. grab a corner if available"
            return True
        
        else:
            print "   Random Move"
            return self.random_move(this_game) # try for any random move (will return True unless the game is a draw because all cells are occupied
            
    def random_move(self, this_game): # pick a random move for the active player... keep trying if the cell is actively filled, unless the game is over
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a random move - This game is a DRAW" % player
            return False
        
        position = random.sample(self.empty, 1) # this position will be available by definition (unless we have an internal error)
                    
        if (self.set_position(position[0], player, player)): # set this position(removed from set) to the current player using player as the token - return True if successful
            print "Found a random move"
            return True #indicates a successful update
            
    def corner_move(self, this_game): # pick a random move for the active player... keep trying if the cell is actively filled, unless the game is over
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a move - This game is a DRAW" % player
            return False
        
        # try to grab a corner, if available
        corners_available = self.corners.intersection(self.empty) # get the available corners
#        print "Corners available = ", corners_available
        if (len(corners_available)!=0): # grab any corner:
            #random_corner = random.randint(1,len(corners_available))

            #some_corner = corners_available.pop()
            some_corner = random.sample(corners_available, 1)[0]
            #print "corner_move() picked random corner ", some_corner, " from potential of ", len(corners_available), " corners."
#            print "Grab corner position %d" % some_corner
            if (self.set_position(some_corner, player, player)): # set this cell to the current player - return True if successful
                return True #indicates a successful update
        
        return False # no obvious offensive move to make
            

    def center_move(self, this_game): # pick a random move for the active player... keep trying if the cell is actively filled, unless the game is over
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a move - This game is a DRAW" % player
            return False
        
        rnum = (self.dim+1)/2 # grab the center position, or near to it
        cnum = rnum
        
        # try to grab the center position
        if (self.set_row_col(rnum, cnum, player, player)): # set this cell to the current player (use the player as the token)- return True if successful
            return True #indicates a successful update
        
        # else just try a defesnive move - return T/F if a move was made
        elif (self.defensive_move(player)): # see if a defensive move is required to block a pending potential win by the player on the next turn - or to finish a pending win for this player
            return True # move was successful
        
        return False # no obvious offensive move to make
            
    def defensive_move(self, this_game): # pick a random move for the specified player... if no obvious defensive moves, return False so another strategy can be attempted
        player = this_game.board.players.whosturn()
        if (self.is_full_board()):
            print "%s cannot make a defensive move - This game is a DRAW" % player
            return False

        # see if the opponent has almost filled up any particular row, column or diagonal - with one empty space remaining.
        # if so, focus on filling the empty cell in that nearly full row, column, or diagonal
        
        # try to grab last space in any nearly filled rows
        for row, row_set in self.rows.iteritems():

            what_is_left = row_set.intersection(self.empty)  #find out which empty cells are still in this row
            
            if (len(what_is_left)==1): # only one spot left in this row... see if I can win this row, or if I have to defend the last spot
                
                what_i_have = row_set.intersection(player.positions_set) #get a set of what I have in this row
                
                if (len(what_i_have)==(self.dim-1)) or (len(what_i_have)==0): #if I have all but one spot, or if I have no spots, I should take this last spot
#                    print "Grabbing last available cell in Row %d which is %s" % (row, what_is_left)                
                    my_status = self.set_position(what_is_left.pop(), player, player) # pop the last member of this intersection set and use it to set the position for the specified player
                    if (my_status == False):
                        print "defensive_move: Internal Error!!! This should not happen"
                    print "Filled last spot on a row"
                    return my_status #True if the position was set correctly
                
            # else more than one space left... keep looking
        
        # try to grab last space in any nearly filled col
        for col, col_set in self.cols.iteritems():

            what_is_left = col_set.intersection(self.empty)  #find out which empty cells are still in this col
            
            if (len(what_is_left)==1): # only one spot left in this col... see if I can win this col, or if I have to defend the last spot
                
                what_i_have = col_set.intersection(player.positions_set) #get a set of what I have in this col
                
                if (len(what_i_have)==(self.dim-1)) or (len(what_i_have)==0): #if I have all but one spot, or if I have no spots, I should take this last spot
#                    print "Grabbing last available cell in Col %d which is %s" % (col, what_is_left)                
                    my_status = self.set_position(what_is_left.pop(), player, player) # pop the last member of this intersection set and use it to set the position for the current player
                    if (my_status == False):
                        print "FAIL!!!"
                    print "Filled last spot on a column"
                    return my_status #True if the position was set correctly
        
        # process all diagonals on the game board
        for diag_num, diag_set in self.diags.iteritems():
        
            #try to grab the last space in a nearly filled diagonal
        
            what_is_left = diag_set.intersection(self.empty)  #find out which empty cells are still in this diag
        
            if (len(what_is_left)==1): # only one spot left in this diag... see if I can win this col, or if I have to defend the last spot
                
                what_i_have = diag_set.intersection(player.positions_set) #get a set of what I have in this diag
                
                if (len(what_i_have)==(self.dim-1)) or (len(what_i_have)==0): #if I have all but one spot, or if I have no spots, I should take this last spot
    #                print "Grabbing last available cell in Diag %d which is %s" % (diag_num, what_is_left)                
                    my_status = self.set_position(what_is_left.pop(), player, player) # pop the last member of this intersection set and use it to set the position for the current player
                    if (my_status == False):
                        print "FAIL!!!"
                    print "Filled last spot on a diagonal"
                    return my_status #True if the position was set correctly
                    
        return (False)  # we could not find any defensive moves



    # determine if the player designated by the single character player string has won the game provided by the >= 9 position TTT game board
    def is_win(self, player): #works  for N dimensional games
        
        # look for a full row of the same player's object instance - that indicates a win for that row
        # check each row on the board
        
        for i in self.rows:
            if self.rows[i].issubset(player.positions_set): #every cell member of set rows[i] is contained in this player's currently occupied cells.. that's a row-win
                print "Player %s Wins!!! (row %d)" % (player, i)
                return True
            
        for i in self.cols:
            if self.cols[i].issubset(player.positions_set): #every cell member of set rows[i] is contained in this player's currently occupied cells.. that's a row-win
                print "Player %s Wins!!! (col %d)" % (player, i)
                return True
        
        for i in self.diags:
            if self.diags[i].issubset(player.positions_set): #every cell member of set diag1 is contained in this player's currently occupied cells.. that's a dia1-win
                print "Player %s Wins!!! (diag, %d)" % (player, i)
                return True
                                
        return False #specified player has not won
        
    
    
######### END TTT Board class

  
# play TTT - pass in Maximum wins & automated status per player

def TTT(MaxWins, p0, p1, doAutomated):

  currentWins = 0

  play_again = 'y'

  while play_again[0] == 'y': # keep going until some player indicates 'n'=no more games
    
    print "\n***************************************************************"
    print "Total Wins so far this round p1 / p0 = ", p0.win_count, "/", p1.win_count, " out of a maximum of ", MaxWins
    print "***************************************************************\n"
    
      
    if ((p0.win_count >= MaxWins) and (p1.win_count>= MaxWins)):
      print "Reached maximum wins... quitting"
      break

    
    print "\nNEW GAME..."

    # init players in case we have looped (soft reset preserves AI learning history)
    p0.reset_soft()
    p1.reset_soft()
    
    # create a n by n tic tac toe board with the provided tuple of players.
    # pass in the functional object we will use to represent the game board (derives from gameworks.Board)
    
    flip_coin = random.randint(1,100) # randomly choose who goes first
    #flip_coin = 100 # p0=X always goes first
    
    if (flip_coin < 50):
      print "** Coin flip = HEADS - Player ", p0, " goes first"
      this_game = gameworks.Game(3, (p0, p1), tic_tac_toe_Board)
    else:
      print "** Coin flip = TAILS - Player ", p1, " goes first"
      this_game = gameworks.Game(3, (p1, p0), tic_tac_toe_Board)
    
    print this_game
    
    this_game.board.moves.append(this_game.board.players.turn); # record the ordinal id of the first player to move in the moves array
    
    #this_game.board.init_board([1, 0, 0, 0, 0, 0, 0, 0, -1], this_game.board.players.players[0], this_game.board.players.players[1])
    
    while True: # keep taking turns until someone wins and breaks out of this loop

        if (not doAutomated):
          predict2(this_game, doSilent=False)         # predict next best move printed out to screen as FYI
        
    # alternate moves for each active player via the next() method invoked below
    
        print "Player %s\'s move..." % this_game.board.players.whosturn()
        

        # request a move for the current player

        if (this_game.board.players.whosturn().automated): #if the currently actively player is automated (not manual), pick a move for them
            if (doAutomated):
              move_status = this_game.board.random_move(this_game) #player an available move in a 'defensive' way
            else:
              move_status = this_game.board.ai_move(this_game) 
        
        else: #manual user input for current player
            my_row = raw_input('Player %s: row number of your choice (1-%d):' % (this_game.board.players.whosturn(), this_game.board.dim))
            if (len(my_row)==0):
                continue # fail - try again via next loop iteration
            my_col = raw_input('Player %s: col number of your choice (1-%d):' % (this_game.board.players.whosturn(), this_game.board.dim))
            if (len(my_col)==0):
                continue
            
            # make the move.  send in the player object for the currently active player. send in that player's individual token representation to be placed on the board (in other games it might be a letter, but for ttt we just use the player's personal token)
            move_status = this_game.board.set_row_col(int(my_row), int(my_col), this_game.board.players.whosturn(), this_game.board.players.whosturn().me_token)
        
        # check validity of selected move

        if (move_status==True): #if the move was valid...
          print this_game.board #print the updated game status
          if (this_game.board.is_win(this_game.board.players.whosturn())): #iswin() prints winner message
              # AI WIP - record this win as a training sample for AI
              # run the AI iterator to update based on this sample
              
              if (this_game.board.players.whosturn().win_count <int(MaxWins)):
                this_game.board.players.whosturn().win_count += 1
                
                print "updated player", this_game.board.players.whosturn(), " win count to = ", this_game.board.players.whosturn().win_count
                
                currentWins += 1 # increment number of wins so far
                
                while (len(this_game.board.moves) < (this_game.board.dim*this_game.board.dim+1)):
                  this_game.board.moves.append(-1) # pad out the moves array to indicate unused positions on the board
                  
                this_game.board.moves.append(this_game.board.players.turn) # append the ordinal ID# of the winning player to the moves array
                
                Winning_moves.append(this_game.board.moves) # append current game moves in the board's history of won games for this player
                
                temp_board = this_game.board.as_array()
                temp_board.append(0) # we append next player's turn id, but in this case the game is won, so the id is 0 (neither -1 or 1)
  
  
                # PLAYER 0 WIN?
                if (this_game.board.players.players[0] == this_game.board.players.whosturn()):
                  print "PLAYER", this_game.board.players.players[0], " WENT FIRST and WON!"
                  
                  # if this board is already in the collection of winning boards for this player, do not add it
                  found_duplicate = False
                  for thisList in Winning_boards_playerA:
                    if (thisList == temp_board):
                      print "duplicates: \n", thisList, "\n", temp_board
                      #raw_input("skip duplicate winning board!")
                      found_duplicate = True
                      break
                  
                  if (not found_duplicate):
                    Winning_boards_playerA.append(temp_board)
                    Winning_boards_all.append(temp_board)
                    Winning_players_all.append(0)  # the player that went first one this game
                    Winning_playerA.append(1) # 0 means player -1 (1st player)
                    #with open("winning_playersA.txt", "a") as myfile:
                    #  myfile.write( "0,\n")
  
                # ELSE PLAYER 1 WIN
                else:
                  print "PLAYER", this_game.board.players.players[1], " WENT SECOND and WON!"
                  
                  # if this board is already in the collection of winning boards for this player, do not add it
                  found_duplicate = False
                  for thisList in Winning_boards_playerB:
                    if (thisList == temp_board):
                      print "duplicates: \n", thisList, "\n", temp_board
                      #raw_input("skip duplicate winning board!")
                      found_duplicate = True
                      break
                    
                  if (not found_duplicate):
                    Winning_boards_playerB.append(temp_board)
                    Winning_boards_all.append(temp_board)
                    Winning_players_all.append(1) # the player that went second as in (0, 1) won this game
                    Winning_playerB.append(1) # 1 means player 2 (2nd player)
                    #with open("winning_playersB.txt", "a") as myfile:
                    #  myfile.write( "1,\n")

                
              #print "Winning board: \n", temp_board
                
              #print "AI WIP: record that game as a win for player %s" %this_game.board.players.whosturn()
              #print "this game has been won with the following moves: \n", Winning_moves
              
              if not this_game.board.players.whosturn().automated:
                  play_again = 'n' # assume no unless we get a y
                  play_again = raw_input('\nPlay again?(y/n):')
                  
              break # now break out of game turn loop - allow outter loop to deal with next game (it will be yes by default until changed)
            
          else:  # not a win.. advance to next player
              this_game.board.players.next() # move the the next player's turn
        
        # else: that position is occuppied, so loop around and let the active player try again
        
        if (this_game.board.is_full_board()): # but first check to see if the game is a draw
            print "%s cannot make a move - This game is a DRAW" % this_game.board.players.whosturn()
            break
          
# END TTT main function

# test bed

def test_bed(players, board_list, winners_lists):
  
  #print "test_bed() here with players: ", players, " and board_list ", board_list
  
  error_count = 0

  for p in range(0,len(players)):
    thisPlayer = players[p]
    
    for b in range(0,len(board_list)):
      thisBoard = board_list[b]
      thisWinner = winners_lists[p][b]
      temp_board = []
      
      for i in thisBoard: # copy the current  board to a temp array
        temp_board.append(i)
        
      #temp_board.append(0) # append id of player who will go next as the last number in the array
        
      #print "thisBoard\n", thisBoard
      #print "thisWinner\n", thisWinner
      #print "i\n", i
      #print "thisPlayer.syn[0]", thisPlayer.syn[0]
      #print "thisPlayer.syn[1]", thisPlayer.syn[1]
      
      l1 = nonlin(np.dot(temp_board,thisPlayer.syn[0])) # vector * matrix = vector
      
      #print "predict2() l1 = \n", l1
      
      dp_result = np.dot(l1,thisPlayer.syn[1])
      
      #print "predict2() dp_result for l2 before nonlin() = \n", dp_result
      
      l2 = nonlin(dp_result) # vector . vector = scalar
      
      #print "predict2() l2 = \n", l2
      
      #print "test_bed(): For player %s the winning probability is %d (vs %d) with board  %s" % (thisPlayer, l2, thisWinner, temp_board)
      
      if (np.abs(thisWinner - l2) > .001):
        error_count += 1
        print "test_bed()  Error for player %s with winning probability is %f (vs %f) with board  %s" % (thisPlayer, l2, thisWinner, temp_board)
        
  print "test_bed(): found ", error_count, "errors"


###################
# global variables

Default_training_sample_count = 5 # how many training samples we will use when needed, unless otherwise specified on the command-line iwth -t #
Default_training_batch_count = 5 # how many training samples we will use when needed, unless otherwise specified on the command-line iwth -t #

# create two players

# automate player 1 if we are doing training, else interactive (p1 is always automated)

p0 = gameworks.Player("x", isAutomated=True) #player 0 represented by an x on the game board and will/will not be automated
p1 = gameworks.Player("y", isAutomated=True) #player 1 represented by an y on the game board and will/will not be automated


Winning_moves = [] # list of lists: where each list is the moves (by linear position #) that were played to win the game - used for AI trainin. Note that first # is ordinal indication of if the winning player went first (0), second(1), etc. and last number is indication of ordinal position of winner WIP - what to do to pad out short games
Winning_boards_playerA = [] # string list of array formatted wnning board layouts... can be used for AI
Winning_boards_playerB = [] # string list of array formatted wnning board layouts... can be used for AI
Winning_boards_all = Winning_boards_init # string list of all winning boards - includes players 0 and 1

Winning_playerA = [] # not currently used
Winning_playerB = [] # not currently used
Winning_players_all = Winning_players_init_playerB

doAutomated = True

# process command-line args - force training or not

if (len(sys.argv) > 1 and (sys.argv[1]=="-t")): # user wants to force training...
  if (len(sys.argv) > 3): # user specified a sample count
    print "\n!!!!!!!!!!\n\nForcing training per command-line -t option, with ", int(sys.argv[2]), "trials, and batch size = ", int(sys.argv[3])
    TTT(int(sys.argv[2]), p0, p1, doAutomated) # play a few automated games to generate some weights based on random wins
    Default_training_batch_count = int(sys.argv[3])
  elif (len(sys.argv) > 2):
    print "\n!!!!!!!!!!\n\nForcing training per command-line -t option, with ", int(sys.argv[2]), "trials...."
    TTT(int(sys.argv[2]), p0, p1, doAutomated) # play a few automated games to generate some weights based on random wins
  else:
    print "\n!!!!!!!!!!\n\nForcing training per command-line -t option, with ", Default_training_sample_count, "trials...."
    TTT(Default_training_sample_count, p0, p1, doAutomated) # play a few automated games to generate some weights based on random wins
  
elif (len(p0.syn) == 0) or (len(p1.syn) == 0):
  print "\n!!!!!!!!!!\n\nForcing training with", Default_training_sample, "samples, due to lack of pre-existing training weights for one or both players..."
  TTT(Default_training_sample_count, p0, p1, doAutomated) # play a few automated games to generate some weights based on random wins
  

# prep data sets based on winning boards and players

while True:
  
  # if we know we need to do training at the start of each loop...
  
  if ( (len(p0.syn) == 0) or (len(p1.syn) == 0)  # if either player does not have pre-existing weights from file (on creation), retrain both
    or len(sys.argv) > 1 and sys.argv[1]=="-t"): # or program was invoked with -t to force recalc of training weights

    print "Boards before training:"
    print "*************************"
    #print "Winning boards for Player A"
    #print Winning_boards_playerA
    #print "Wining players for A"
    #print Winning_playerA
    
    #print "\nWinning boards for Player B"
    #print Winning_boards_playerB
    #print "Wining players for B"
    #print Winning_playerB
    
    print "\nWinning boards for all"
    print Winning_boards_all
    print "Wining players for all"
    print Winning_players_all
    print "*************************"
    
    # add losing situations
    
    #for thisBoard in Losing_boards_init:
    #  Winning_boards_all.append(thisBoard)
    
    # prep data sets based on winning boards and players
  
    # train using recent wins as training data
    
    Winning_players_all_C = []
    for wp in range(0,len(Winning_players_all)):
      Winning_players_all_C.append(1-Winning_players_all[wp])
    
    #for wp in Losing_players_init: # both players should lose in this situation... these are losing boards
    #  Winning_players_all.append(wp)
    #  Winning_players_all_C.append(wp)
  
    X = np.array(Winning_boards_all)
    y = np.array([Winning_players_all]).T # probability of wins for player B (aka player 1, the second player)
    
    #yC = 1-y # probability of wins for player A (aka player 0, the first player)
    yC = np.array([Winning_players_all_C]).T # compliment
    
    print "Player A(0) wins: yC = \n", yC
    print "Player B(1) wins: y = \n", y
  
    print "\n\nOne moment while I retrain player 0..."
    print "---------------------------------------------"
    do_train(X, yC, p0, Default_training_batch_count) # train for player 0
    
    print "\n\nOne moment while I retrain player 1..."
    print "---------------------------------------------"
    do_train(X, y, p1, Default_training_batch_count) # train for player 1
    
    print "DONE TRAINING\n"
    
    print "Weights after training:"
    print "***********************"
    
    print "main:p0.syn = ", p0.syn
    print "main:p1.syn = ", p1.syn
  
  test_bed([p0, p1], Winning_boards_all, [Winning_players_all_C, Winning_players_all])

  
  # now play, using the recently calculated syn0, syn1 weights from above..
  
  p0.automated = False # allow player 0 (the user) to choose each move.  p1 will still be automated
  p1.automated = True # have the computer control this player.
  p0.win_count = 0
  p1.win_count = 0
  
  doAutomated = False # stops random selection of moves by automated players... p1 will now use the trained AI instead of random moves
  
  print "\n*********************"
  print "OK, let's play !!"
  print "*********************\n"
  
  TTT(5, p0, p1, doAutomated) # p0 is manual, p1 is automated
  
  # note above that winning arrays will be augmented for the next loop
