#!/usr/bin/env python
import random
import pickle
import os.path

# THIS MODULE:
# implement a generalized NxN game board that is stored as a linear dictionary of values, but which can be access also via a row/col format
# provide ability to store a "move" per board "cell", where each "move" is owned by a specific player, and may optionally include an object payload (a letter, a symbol, etc.)
# for games like tic tac toe, the payload is just the player's token indicator (x or o, etc.).  For words with friends, this would be the specific letter played by the the player 


#represents one space on the board.
# Has a player object and an 'token' object..
# this 'token' we place on the cell.. which may just be a player indicator (as in tic tac toe) or it could be a letter as in scrabble

class Cell(object):

    def __init__(self, player, token): #token can just be the player object if there is no specific token being played by the player (as in tic tac toe)
        self.claimed_by = player
        self.token_played = token
        
    def __str__(self, ):
        return(str(self.token_played)) # return the printable string representation of the player's move at this location.

# create a game token - could be a letter with a point value and an id, or could be a representation of the player for placement on the board
# requires its own hash so that we can use it in sets

class Token(object):
    
    def __init__(self, token_char, token_value, token_id):
        self.char = token_char # used to display the token - does not need to be unique
        self.value = token_value # used for scoring the token
        self.id = token_id # guaranteed unique ID - helpful for duplicate instances of the same token's display character
        
    def matches(self, this_char): #return true if the provided single character matches the character value for this token
#        print "this_char = %s, self.char=%s" % (this_char, self.char)
        return (this_char.lower() == self.char.lower())
    
        
    def __repr__(self, ):
        return("("+self.char+", "+str(self.value)+", "+str(self.id)+"), ") #use the token id only for comparing during set operations
    
    def __eq__(self, other):
        if isinstance(other, Token):
            return ((self.id == other.id))
        else:
            return False
        
    def __ne__(self, other):
        return(not self.__eq__(other))
    
    def __hash__(self, ):
        return self.id
    
    def __str__(self, ):
        return(self.char)

    
# create the game board

class Board(object):
    
    #   initialize the game board to the specified dimension by creating a NxN game board and initializing the empty set so we can track avaialble cells during game play
    def __init__(self, dimension, tuple_of_players): # will be extended by subclasses 
        self.board = {} # dictionary of possible states for every cell of the game = either player objects or the empty character
        self.dim = dimension # dimension of the game - dim=3 means 3x3 = 9 cells is standard
        self.empty = set() # create a set we can use to track board cells which are currently empty. These are board positions (equivalent of row/col pairs)

        for i in range(1, dimension*dimension+1): #initialize the board to 'empty'
            #print "init board position = ", i
            self.empty.add(i) #add this position to the set of currently known empty cells
            
        self.players = Players(tuple_of_players) # send in a tuple of the player objects to create a convenient container for cycling turns

        self.moves =[] # array of positions which indicate the order in which board positions were filled

    #print out the NxN game board by printing the player token at each occupied cell, or an empty character indicator
    # may be over-ridden by specific game __str__ implementation
    def __str__(self, ):
#        print "empty:", self.empty
        mystring =""
        for row in range(1,self.dim+1):
            for col in range(1,self.dim+1):
                
                position = self.get_position_num(row, col) #get the linear position of this cell on the board
                
                if (self.board.has_key(position)): #there is an existing move at this location
                    this_cell = self.board[position] #get the contents of the cell
                    mystring += str(this_cell) # convert (print) the contents of the cell to a string
                else:
                    mystring += "-" # just fill in with an "empty" cell character
                    
                if (position%self.dim==0): # end of row, hit return
                    mystring += "\n"
            
        return mystring

    # return the current board positions as an array
    def as_array(self, ):
#        print "empty:", self.empty

        my_array = []
        
        #mystring ="\n"
        
        for row in range(1,self.dim+1):
            for col in range(1,self.dim+1):
                
                position = self.get_position_num(row, col) #get the linear position of this cell on the board
                
                if (self.board.has_key(position)): #there is an existing move at this location
                    this_cell = self.board[position] #get the contents of the cell
                    
                    if (self.players.players[0] == this_cell.claimed_by):
                      #mystring += str(-1) # convert (print) the contents of the cell to a string
                      my_array.append(-1)
                    elif (self.players.players[1] == this_cell.claimed_by):
                      #mystring += str(1)
                      my_array.append(1)
                else:
                    #mystring += "0" # just fill in with an "empty" cell character
                    my_array.append(0)
                
                #mystring += ", "
                    
                #if (position%self.dim==0): # end of row, hit return
                #    mystring += "\n"
                    
        
        #mystring += ""
        
        return my_array
        #return mystring

    # set the initial state of the board based on a provided array of -1, 0, 1 values (p0, no move, p1)
    # remember that board positions in the board object are 1-based, while the ordinal positoins of the provided board_state are zero based
    def init_board(self, board_state, p0, p1):
#
        
      #print "init_board() here with ", board_state, " and players ", p0, " and ", p1
      #print "    with current board state before init = ", self.board
      #print "    with empty positions = ", self.empty
      
      #mystring ="\n"
      
      for p in range(0, len(board_state)):
        #print "      new board value would be = ", board_state[p]
        if (board_state[p]==0):
          continue
        elif (board_state[p]==-1):
          self.set_position(p+1, p0, p0.me_token) # p+1 is needed to map between ordinal board_state values and the board's dictionary 1-based values
        elif (board_state[p]==1):
          self.set_position(p+1, p1, p1.me_token)
          
        #self.set_position(p, player, token) #set the linear position of this cell on the board
      
      #print "    board state after init = ", self.board
      return


    # indicate if the requested position on the game board is occupied
    def is_empty_position(self, rnum, cnum):# return true if the requested game board position is currently unoccupied
        requested_cell = set([self.get_position_num(rnum, cnum)]) # create a set with the requested positions's linear position on the game board
        
        # if the requested cell is a subset of the set of empty squares on the game board, then we know that the requested cell is empty
        
        if (requested_cell.issubset(self.empty)): # confirm that this cell is unoccupied
            return True #indicates cell is empty
        else:
            return False

    # see if we have reached a draw since there are no empty spaces left         
    def is_full_board(self, ): 
        if (len(self.empty) == 0): #no more empty slotsin the "empty" set.
            print "This game is a DRAW!! (set empty)"
            return True
        else:
            return False

    # see if the entire game board is empty - noone has played a token yet         
    def is_empty_board(self, ): 
        if (len(self.board) == 0): #no Cell objects have been placed on the board yet..this is a new game
            print "NEW GAME"
            return True
        else:
            return False

    # determine if the provided list of row/column tuples is colinear in a row or column [(r1,c1), (r2,c2)....]
    def is_colinear(self, row_col_tuples):
        
         # see if all row/col tuples are on the same row
        same_rows = True # assume true until we determine otherwise
        row = row_col_tuples[0][0] # get row number of first tuple
        for rc in row_col_tuples:
            if rc[0] != row:
                same_rows = False
                break
        
        # see if everything all row/col tuples are on the same column
        same_cols = True # assume true until we determine otherwise
        col = row_col_tuple[0][1] # get col number of first tuple
        for rc in row_col_tuples:
            if rc[1] != col:
                same_cols = False
                break
            
        if same_cols or same_rows: # if either in same row or same column, then the row col tuples are colinear
            return True
        else:
            return False
    
    # determine whether the current player's partial move on the board is valid or not.
    # if valid return True = means the "play" can be finished
    # if not valid return False = means the "play" is invalid and the player must recall/reassess their turn, or pass
    # this method will be over-written by each individual game board since each game has specific rules
    def is_valid_play(self, ):
        return NotImplemented 
                
    # current player is ready to submit their current tokens on the board as an official move
    # relevent for multi-cell plays as in words with friends, scrabble, battleship, etc
    def play(self, ):
        
        if (self.players.whosturn().partial_move() == False): # there is no move on the board for this player, return false
            print "   Play Error - cannot submit move as there are no tokens for %s on the board" % self.players.whosturn()
            return False
        
        if (self.is_valid_play() == False): # validity check for this players status on the board is bad
            print "   Play Error - token placement is invalid.  Please recall and try again"
            return False
        
        
        #WIP we will have to do a thorough validty check here to ensure the tokens in play are "valid" for each specific game
        # WIP whihc may require extension of this method by specific game implementations.  Or separation of validation from "play()"

        self.players.whosturn().pass_count = 0
        self.players.whosturn().passed = False
        return True # WIP return true or false to indicate if the current play was valied


    #this player would like to recall their current tokens on the board prior to an official move.
    #Take them off the board, and put them back in the pool of the player's tokens
    def recall_tokens(self, player):

        for token_tuple in player.tokens_in_play:
            print "   Recalling token", token_tuple
            
            token_position = self.get_position_num(token_tuple[0], token_tuple[1])
            
            player.tokens_set.update([token_tuple[2]]) # add each token from the tuple back to the player's token set as we are recalling them from the board

            # remove this board position from the player's list of cells s/he has occupied on tht board
            player.positions_set.difference_update([token_position])     # remove the equivalent board position for this row/col specified from the player's list of owned cells
            
            # add this board position back to the empty list for the board
            self.empty.update([token_position])
            
            # delete this Cell from the board dictionary
            del self.board[token_position]
            

        del player.tokens_in_play[:] # deleate all tokens in this list as we have put them back in the player's inventory
        
    # replentish the specified player's token so they have max_token tokens on hand.
    # typically called after a player "plays" the tokens they have already placed on the board, assuming it was a valid play
    def replentish_max(self, player, max_tokens):
    
        token_count = max_tokens - len(player.tokens_set) # determine how many tokens to give based on what the player currently has on hand vs the max # provided
       
        if (len(self.all_game_tokens_set)==0): # we cannot replentish as there are no tokens left in the game stache
           print "   No letters left in Game Cache to replentish this player's inventory!"
       
        elif (len(self.all_game_tokens_set)<token_count): # game has tokens left, but not enough to replentish the amount just used.  So take what's left.
           replentish_tokens = random.sample(self.all_game_tokens_set, len(self.all_game_tokens_set))  # randomly select N tokens from the set of all available tokens and give it to the player for their next turn
           self.all_game_tokens_set.difference_update(replentish_tokens)  # remove those tokens from the game's list of available tokens WIP - this could be cleaner here - just reset self.all_game_tokens_set to None
           player.tokens_set.update(replentish_tokens) # add those tokens to the player's inventory
           print "   Player %s received new letter(s) %s (%s left in game)" % (player, replentish_tokens, len(self.all_game_tokens_set))
           
        else: # we have enough tokens.. replace amount used
           replentish_tokens = random.sample(self.all_game_tokens_set, token_count)  # randomly select N tokens from the set of all available tokens and give it to the player for their next turn
           self.all_game_tokens_set.difference_update(replentish_tokens)  # remove those tokens from the game's list of available tokens
           player.tokens_set.update(replentish_tokens) # add those tokens to the player's inventory
           print "   Player %s received new letter(s) %s (%s left in game)" % (player, replentish_tokens, len(self.all_game_tokens_set))

        del player.tokens_in_play[:]  # delete all tokens in this list as we are submitting the play
#        print "tokens in play after delete", player.tokens_in_play
        return True

    def extends_a_cell(self, rnum, cnum): # WIP to be used to determine if a set of tokens in play extend an existing cell or are floating in the middle (illegal)
        pass
    

    # convert a row/col pair into an absolute position on the linear game board 1..dim^2
    def get_position_num(self, rnum, cnum):
        if (rnum <1 or rnum > self.dim or cnum<1 or cnum>self.dim):
            print "   Try again - row/column numbers must be from 1 to %d" % self.dim
            return None
        return(self.dim*(rnum-1)+cnum)

    # convert an absolute position on the linear game board 1..dim^2 into a row/col pair 
    def get_row_col(self, position):
        if (position <1 or position > self.dim*self.dim):
            print "   get_row_col(): Try again - position numbers must be from 1 to %d" % self.dim*self.dim
            return None
        return(int((position-1)/self.dim)+1, ((position-1)%self.dim)+1)

    #set 1 or more positions on the board within a given row, starting at a given r/c location, for every token provided (1 or more), left to right
    # tokens is a list of tokens to be placed on the board
    # return True if successful, or fall if the requested spaces are not all free, or if the set of tokens goes beyond the end of a row
    # only place all the tokens if all of the spaces are available (no partial moves)
    def set_row_positions(self, rnum, cnum, player, tokens):

        new_words = [] # empty list we will fill with words that are created by this play
        
        if rnum<1 or rnum > self.dim or cnum<1 or cnum>self.dim:
            print "   Selection out of bounds.. please try again. Values must be between 1 and %s" % self.dim
            return False

        # get position of rnum cnum
        position = self.get_position_num(rnum, cnum)

        # convert the requested spaces into a list of absolute positions
        length_of_run = len(tokens)
        
        #make sure we do not have too many tokens to place on this row - must go on the same row!
        if (cnum+length_of_run-1 > self.dim): #cannot place all tokens on this row - error
            print "   Try again - not enough room to place %s on row %d starting at column %d" % (tokens, rnum, cnum)
            return False
        
        # create a set of all positions being requested
        spots_to_fill = set()
        for i in range(position, position+length_of_run):
#            print "check if spot %d is empty" % i
            spots_to_fill.add(i)
          
        if spots_to_fill.issubset(self.empty):  # all of these positions are available for placement
#            print "Debug: place %s at %d %d on row %d" % (tokens, rnum, cnum, rnum)
            for i in range(length_of_run):
#                print "fill position %d with %s" % (position+i, tokens[i])
#                self.set_position(position+i, player, tokens[i])
                self.set_row_col(rnum, cnum+i, player, tokens[i])
#                player.tokens_in_play.append((rnum, cnum+i, tokens[i])) #apped the tuple row/col/token to the list of tokens which are actively in place.  This list will be cleared by the player.play() function if the move is valid

            this_word = self.get_row_words(rnum, cnum)
            if len(this_word)>0:
                new_words.append(this_word) # get the words on this row at this col (or the nearest word starting to its left) to help us validate that these token placements generate correct works
            for i in range(length_of_run):
                this_word = self.get_col_words(rnum, cnum+i)
                if len(this_word)>0:
                    new_words.append(this_word) # make sure that every row that contains a letter from this token set is a valid word (i goes form 0 to len-1)
            print "   New words found on this move:", new_words
            return True
        else:
            print "   Try again - The requested spaces at rc = %d %d on row %d are not available for %s" % (rnum, cnum, rnum, tokens)
            return False
    
    #set 1 or more positions on the board within a given column, starting at a given r/c location, for every token provided (1 or more), left to right
    # tokens is a list of tokens to be placed on the board
    # return True if successful, or fall if the requested spaces are not all free, or if the set of tokens goes beyond the end of a column
    # only place all the tokens if all of the spaces are available (no partial moves)
    def set_col_positions(self, rnum, cnum, player, tokens):
        new_words = [] # empty list we will fill with words that are created by this play
        
        if rnum<1 or rnum > self.dim or cnum<1 or cnum>self.dim:
            print "   Selection out of bounds.. please try again. Values must be between 1 and %s" % self.dim
            return False

        # get position of rnum cnum
        position = self.get_position_num(rnum, cnum)

        # convert the requested spaces into a list of absolute positions
        length_of_run = len(tokens)
        
        #make sure we do not have too many tokens to place on this col - must go on the same col!
        if (rnum+length_of_run-1 > self.dim): #cannot place all tokens on this row - error
            print "   Try again - not enough room to place %s on column %d starting at row %d" % (tokens, cnum, rnum)
            return False
        
        # create a set of all positions being requested
        spots_to_fill = set()
        for i in range(position, position+length_of_run*self.dim, self.dim): # since we are moving along columns we need to step by "dim" positions on each iteration
#            print "check if spot %d is empty" % i
            spots_to_fill.add(i)
          
        if spots_to_fill.issubset(self.empty):  # all of these positions are available for placement in this column
#            print "Debug cols: place %s at %d %d on col %d" % (tokens, rnum, cnum, cnum)
            for i in range(length_of_run): # place the tokens on the board in this column
#                print "fill position %d with %s" % (position+i*self.dim, tokens[i])
#                self.set_position(position+i*self.dim, player, tokens[i])
                self.set_row_col(rnum+i, cnum, player, tokens[i])
#                player.tokens_in_play.append((rnum+i, cnum, tokens[i])) #apped the tuple row/col/token to the list of tokens which are actively in place.  This list will be cleared by the player.play() function if the move is valid
                
            this_word = self.get_col_words(rnum, cnum)
            
            if len(this_word)>0:
                new_words.append(this_word)  # get the words on this column to help us validate that these token placements generate correct words
                
            for i in range(length_of_run):
                this_word = self.get_row_words(rnum+i, cnum)
                if len(this_word)>0:
                    new_words.append(this_word) # make sure that every row that contains a letter from this token set is a valid word (i goes form 0 to len-1)
            print "   New words found on this move:", new_words
            return True
        else:
            print "   Try again - The requested spaces at rc = %d %d on col %d are not available for %s" % (rnum, cnum, cnum, tokens)
            return False
     
    
    # set the specified ABSOLUTE position (an integer) on the game board to the specified player and the player's token (just the player in this case - could be a letter, etc in other games)
    # update sets accordingly
    # return True if successful
    # return False if unsuccessful - which would mean that the cell was already occupied
    def set_position(self, position, player, token):

        requested_position = set([position]) # create a set with the requested position integer as its only member
        
        if (requested_position.issubset(self.empty)):  # confirm that this cell is unoccupied because it is in the empty set)
            new_cell = Cell(player, token) # create a new Game cell to store at this position
            self.board[position] = new_cell #store the object reference for currently active player
            player.positions_set.add(position)  #add this current position to the set of cells occupied by this player
            self.empty.remove(position) # remove this position from the list of available cells on the game board
            player.tokens_set.difference_update([token]) # remove this token from the player's inventory (token must be a list)

            # AI WIP - record order of token placement for future assessment
            self.moves.append(position);
            #print "set_position() AI WIP record storage in ", position, " by player ", player, " using token ", token
            #print "set_position() AI WIP moves array = ", self.moves
            
            return True #indicates a successful update
        
        else:
            print "cannot fill this spot", position
            return False

    # set the specified row/column position on the game board to the specified player and that player's token for this specific move
    # update sets accordingly
    # return True if successful
    # return False if unsuccessful - which would mean that the cell was already occupied

    def set_row_col(self, rnum, cnum, player, token ):
        
        if rnum<1 or rnum > self.dim or cnum<1 or cnum>self.dim:
            print "   Selection out of bounds.. please try again. Values must be between 1 and %s" % self.dim
            return False

        position = self.get_position_num(rnum, cnum) #convert rnum cnum into a linear position on the linear board for dictionary lookup
        
        my_status = self.set_position(position, player, token) #call the base function which makes the move
        
        if (my_status == True):  # success
            print "   Player %s places %s on the Board at (%d %d)" % (player, token, rnum, cnum)
            player.tokens_in_play.append((rnum, cnum, token)) #append the tuple row/col/token to the list of tokens which are actively in place.
            #  ... (above) This list will be cleared by the player.play() function if the move is valid
            return True
    
        print "   Game board cell (%d, %d) is occupied.. please try again" % (rnum, cnum)
        return False

    # attempt to claim the board position rnum, cnum for the specified player using the specified token; return True if successful, return False if occupied already
    def move(self, rnum, cnum, player, token): 

        if rnum<1 or rnum > self.dim or cnum<1 or cnum>self.dim:
            print "   Selection out of bounds.. please try again. Values must be between 1 and %s" % self.dim
            return False
        
        if (self.set_row_col(rnum, cnum, player, token)): # set this cell to the current player using the specified token - return True if successful
            return True #indicates a successful update
        else:
            print "   Selection occupied.. please try again"
            return False #indicates the cell requested was already occupied either by the current player or the other player

        
    # return the player or token object or None indcator at the requested position from the game board dictionary
    def get_token_from_board(self, rnum, cnum):
        
        position = self.get_position_num(rnum, cnum)  # convert rnum/cnum into a position
        
        if self.board.has_key(position): # there is a move at this position, return the player object that 'owns' this cell
                return self.board[position]
        else:
            print "get_token_from_board: no token at %d %d yet", rnum, cnum
            return None #there is no owner as noone has claimed this position yet
                        
    # return the word that contains the position row/col, valid or not, in the specified row, starting at the specified column (or the nearest column to the left that is not empty)
    # may return an empty string "" or 1 letter word, if discovered
    def get_row_words(self, row, col):
        
        starting_col = col #assumeto our word starts at row,col - but it may extend an existing word... so we will have to backup to the left on the row (col-1) to find the beginning, which may be the far left of the row

        while (starting_col>1) and not self.is_empty_position(row, starting_col-1): # while we are not at the far left of this row, and the next position to the left is not empty
            starting_col -= 1 # decrement the col number (go left 1 col)
    
        # now we should be at the start of a word that contains row/col
        
        word_letters_so_far = "" # initialize the letters found so far on this row to nothing
        
        #now starting_col should represent the start of a word that includes the row/col position passed to this function either at the left of a row or with a blank position to the left of it
        for this_col in range(starting_col,self.dim+1):  #check every row in this column starting at the 
            position = self.get_position_num(row, this_col)
#           print this_row, col, position
            if (self.is_empty_position(row, this_col)): # we moved beyond the end of a word
                break
            else:
                word_letters_so_far += str(self.get_token_from_board(row, this_col)) # collect the token stored at this position - convert to string and add to end of word string
        
        # return what we have
#        print "row word", word_letters_so_far
        return word_letters_so_far



    # return the word that contains the token stored at position row/col, valid or not, in the specified row, starting at the specified column (or the nearest column to the left that is not empty)
    # may return an empty string "" or 1 letter word
    # assume there is a token at row/col, if not this is an benign error - an empty string will be returned
    def get_col_words(self, row, col):
        
        starting_row = row #assume our word contains the token at row,col - but it may extend an existing word... so we will have to backup the column row by row to find the beginning, which may be the top of the col

        while (starting_row>1) and not self.is_empty_position(starting_row-1, col): # while we are not at the top of this column, and the next position above is not empty
            starting_row -= 1 # decrement the row number (go up 1 row)
    
        word_letters_so_far = "" # initialize the letters found so far on this column to nothing
        
        #now starting_row should represent the start of a word that includes the row/col position passed to this function either at the top of a col or with a blank position above it
        for this_row in range(starting_row,self.dim+1):  #check every row in this column starting at the row identified above and continuing to the end of this column
            
#           print this_row, col, position
            if (self.is_empty_position(this_row, col)): # we moved beyond the end of a word, so break out of the loop - we are done
                break
            else:
                word_letters_so_far += str(self.get_token_from_board(this_row, col)) # collect the token stored at this position - convert to string and add to end of word string
        
        # return what we have so far
#        print "col word", word_letters_so_far
        return word_letters_so_far
                
            
        
class Player(object):
    
    def __init__(self, my_name, isAutomated):
        self.name = my_name # character representing this player
        self.automated = isAutomated #true if we want the computer to move for this player
        self.positions_set = set() #set of all board positions that this player currently occupies - these may be integers that correspond to positions on a linear representation of the game board
        self.tokens_set = set() # set of depleatable tokens that this player has in their position which have not yet been placed on the board
        self.me_token = Token(my_name, 0, isAutomated) # create a token to represent this player on the board - that uses this player's name, has no value, and has an id that equals the player's name string
        self.passed = False # True if this player decided to pass on their most recent turn
        self.pass_count =0 # number of times this player has skipped their turn in a row.
        self.tokens_in_play = []  # list of 3-value tuples represented by (row, col, token object), where 1<=r/c<=15 and token object is a valid player token.  These represent tokens which have been placed on the board, but have not yet been validated by a "play" action from the player
        self.syn = [] # list of ai synapse weights syn0, syn1, etc for this player... supports ai learning
        self.win_count = 0 # count of wins this player has had so far
        
 
        psyn_file_name = str(self.name) + '.syn.txt' 
        print "Player.init(): looking for ", psyn_file_name, "...."
        
        if (os.path.isfile(psyn_file_name)):
          print "Player.init(): Loading previous synapse training for player ", str(self.name), "..."
          with open(psyn_file_name,'rb') as psyn:
            self.syn = pickle.load(psyn)
          print "     DONE! Loaded", len(self.syn), "synapses, representing", len(self.syn[0]), "nodes and", len(self.syn[1]), "training samples"
          print self.syn
        else:
          print "     No synapse file found for player ", self.name, ". Abort Player Synapse Init."

              
        
    def __str__(self, ):
        return(self.name) #print the player's name
    
    def partial_move(self, ):
        if len(self.tokens_in_play)>0: # true if this player has placed tokens on the board but has not yet submitted them via the "play" method
            return True
        else:
            return False # this player has not yet put any tokens on the board as a partial move
        
    def pass_turn(self, ): # make this player skip their turn
        self.pass_count += 1
        self.passed = True
        
    def reset_soft(self, ): # reset game history for this player from previous games - preserve AI info
        self.positions_set = set() #set of all board positions that this player currently occupies - these may be integers that correspond to positions on a linear representation of the game board
        self.tokens_in_play = [] # reset any tokens that would have been in play
        return True
        
    def reset_hard(self, ): # reset game history for this player from previous games - destroy AI info
        self.positions_set = set() #set of all board positions that this player currently occupies - these may be integers that correspond to positions on a linear representation of the game board
        self.tokens_in_play = [] # reset any tokens that would have been in play
        self.syn = [] # reset AI synapse weights
        return True       
    
    def get_token_from_player(self, this_char): #get a token that corresponds to the requested letter.. do not remove from inventory, just return the token.  Return None if not found
        
        for this_token in self.tokens_set: # check for the requested character in this player's inventory of tokens
            if this_token.matches(this_char):
                return this_token
        
        return None # if we have checked all tokens, return false to indicate that the inventory does not contain this letter
        
        
# create a container object for the provided tuple of individual player objects
class Players(object):
    
    def __init__(self, players ): #players is a tuple of multiple Player objects
        self.turn = 0 #ordinal list position of the player who's turn is currently active, player zero is the zero position of the list... usually player 1
        self.players = players # store the tuple of the players we have been given
        
    def whosturn(self, ):
        return self.players[self.turn] # return the object of player who is currenty up to play
    
    def next(self, ):
        self.turn = (self.turn+1) % len(self.players) #advance the ordinal position in the list of players
#        print "next up: player %s" % self.players[self.turn]
        return self.players[self.turn] #return the object of the player who is now up for a turn
        
    def __str__(self, ): #print all players information... put a start next to the name of the player who's current turn is active
        my_string = ""
        for i in range(0,len(self.players)):
            if i==self.turn:
                my_string += "*"
            my_string += str(self.players[i].name)+", "
        return(my_string)
        
class Game(object):  #create a random game object with players, a game board, and an indication of which player's turns should be automated

    def __init__(self, dimension, tuple_of_players, game_type):
        self.board = game_type(dimension, tuple_of_players) #create the game board, pass in the players so they may be given initial inventories as needed
 

    def __str__(self, ):
        return str(self.board)
 