import numpy as np
from numpy import random
from itertools import permutations
import math as m
import matplotlib.pyplot as plt

def minefield(n):
    minefield_ = np.zeros(n**2,dtype = int).reshape(n,n)
    return minefield_

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

def gameboard(n,k):
    mine_list = mine_perm(n)[k]
    board = minefield(n)
    for i in mine_list:
        a = i[0]
        b = i[1]
        board[a,b]=100
    return(board)

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

def initialize(board,i,j):
    board_dash = board.copy()
    board_dash[i,j] = board_dash[(i,j)] + 1
    return board_dash

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

def plot_game_strat_1(n):
    allowed_mines = np.arange(0,m.factorial(n))
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

print(plot_game_strat_1(5))
