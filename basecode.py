import numpy as np
from numpy import random
from itertools import permutations
import math as m
import matplotlib.pyplot as plt

#this just defines a nxn matrix with zeros everywhere
def minefield(n):
    minefield_ = np.zeros(n**2,dtype = int).reshape(n,n)
    return minefield_

#this is a list of the n! different distributions of mines, given the condition that each row and column has exactly one mine
#so each element of this list is a collection of n coordinates where we are going to put the mines
def mine_perm(n):
    list_ = []

    perm_seq = []
    for i in range(0,n):
        perm_seq.append(i)
    perm = list(permutations(perm_seq))

    rows = perm_seq
    for i in range(0,len(perm)):
        columns = perm[i]
        list_temp = []
        for i in range(0,n):
            a = rows[i]
            b = columns[i]
            c = [a,b]
            list_temp.append(c)
        list_.append(list_temp)
    return list_

#this defines your final nxn gameboard with zeros everywhere excpet the mines, which are represented by 100
#the value k specifies which permutation from mine_perm you want to use on the board
#so using k, you basically choose your mine distribution (you have n! different distributions to choose from)
def gameboard(n,k):
    mine_list = mine_perm(n)[k]
    board = minefield(n)
    for i in mine_list:
        a = i[0]
        b = i[1]
        board[a,b]=100
    return(board)

#this is the moving strategy
#i have designed keeping in mind that the objective is to reach the bottom right of the board after initializing at the top left
#say you standing in the middle of the board, you will have 8 adjacent squares that you can see
#the player cant move to a square that has a mine, and ofc it cant move out of the board
#it follows the priority order --> (down + right, down, right, down + left, up + right, left, up, left + up)
#so whichever of the 8 squares are available, it will move to the one which has the highest priority order among them
#whenever the player enters a square you add 1 to its value, and do the same when it leaves a square. so the position of the player is whichever square has an odd value
#using this function on a board only makes one move. if you want five moves you have to use the function five times
def moving_strat_1(board):
    board_dash = board.copy()
    n = board_dash.shape[0]
    for i in range(0,n):
        for j in range(0,n):
            if board_dash[i,j]%2!=0:
                current = i,j
    a = current[0]
    b = current[1]
    list_around = list(((a+1,b+1),(a+1,b),(a,b+1),(a+1,b-1),(a-1,b+1),(a,b-1),(a-1,b),(a-1,b-1)))
    list_around_dash = list_around.copy()
    for i in list_around:
        if i[0]>=n:
            list_around_dash.remove(i)
        elif i[1]>=n:
            list_around_dash.remove(i)
        elif i[0]<0:
            list_around_dash.remove(i)
        elif i[1]<0:
            list_around_dash.remove(i)
        elif board_dash[i]>=100:
            list_around_dash.remove(i)
    next = list_around_dash[0]
    board_dash[current] = board_dash[current] + 1
    board_dash[next] = board_dash[next] + 1
    return board_dash

#this puts your player on the board on the specified coordinates given by i,j
def initialize(board,i,j):
    board_dash = board.copy()
    board_dash[i,j] = board_dash[(i,j)] + 1
    return board_dash

#using this function on a given board will run the moving_strat_1 till the player reaches the bottom right square.
#it gives the final board and the number of steps it has taken to reach the destination
#the different numbers on the board allow you to see the number of times a square has been visited and hence allow you to trace the path it has followed
def game_strat_1(board):
    board_dash = board.copy()
    n = board_dash.shape[0]
    if board_dash[n-1,n-1]>=100:
        return "final square is a mine"
    else:
        counter = 0
        while board_dash[n-1,n-1]%2==0:
            board_dash = moving_strat_1(board_dash)
            counter = counter + 1
        return board_dash,counter

#for a given n, this plots the number of steps taken to reach the destination using moving_strat_1 for different mine distributions on a nxn board
#so we have n! different possible mine distributions but I have removed those distributions in which either the top left or the bottom right square was a mine
#so the x-axis will have a little less than n! values
def plot_game_strat_1(n):
    allowed_mines = list(np.arange(0,m.factorial(n)))
    allowed_mines_copy = allowed_mines.copy()
    for i in allowed_mines:
        if gameboard(n,i)[n-1,n-1]>=100:
            allowed_mines_copy.remove(i)
        elif gameboard(n,i)[0,0]>100:
            allowed_mines_copy.remove(i)
    no_of_steps = []
    for i in allowed_mines_copy:
        a = gameboard(n,i)
        b = initialize(a,0,0)
        c = game_strat_1(b)
        no_of_steps.append(c[1])
    x_axis = np.arange(0,len(allowed_mines_copy))
    plt.bar(x_axis,no_of_steps)
    return plt.show()

plot_game_strat_1(5)