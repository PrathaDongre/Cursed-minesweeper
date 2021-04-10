import numpy as np
import random
import matplotlib.pyplot as plt
import imageio
from pathlib import Path
import pandas as pd
import math

def grid(n):
    grid = np.zeros(n**2,dtype=int).reshape(n,n)
    return grid

def minefield(n,prob):
    field = grid(n)
    for i in range(0,n):
        for j in range(0,n):
            if [i,j] != [0,0] and [i,j] != [n-1,n-1]:
                if random.random()<=prob:
                    field[i,j] = -100
    return field

def initialize(field,i,j):
    field_ = field.copy()
    field_[i,j] += 1
    return field_

def onemove(field):
    field_ = field.copy()
    n = int(field.size**(1/2))
    for i in range(0,n):
        for j in range(0,n):
            if field_[i,j]%2 != 0:
                a = i
                b = j
    current = a,b
    list = [(a-1,b),(a,b-1),(a,b+1),(a+1,b)] #[(a-1,b-1),(a-1,b),(a-1,b+1),(a,b-1),(a,b+1),(a+1,b-1),(a+1,b),(a+1,b+1)]
    list_ = list.copy()
    for i in list:
        if i[0]<0:
            list_.remove(i)
        elif i[0]>n-1:
            list_.remove(i)
        elif i[1]<0:
            list_.remove(i)
        elif i[1]>n-1:
            list_.remove(i)
        elif field_[i] == 2:
            list_.remove(i)
    if list_==[]:
        return False
    elif field_[current] == -99:
        return True
    else:
        next = random.choice(list_)
        field_[current] += 1
        field_[next] +=1
        return field_,next

def run(field):
    n = int(field.size**(1/2))
    field_ = field.copy()
    field_ = initialize(field_,0,0)
    list =[(0,0)]
    while field_[n-1,n-1] == 0:
        new = onemove(field_)
        if new != False and new != True:
            field_ = new[0]
            list.append(new[1])
        elif new == True:
            return "death",list
        elif new == False:
            return "lost",list
    return "success",list

def plot1(field):
    n = int(field.size**(1/2))
    list = np.array(run(field)[1])
    y_values = list[:,0]+0.5
    x_values = list[:,1]+0.5
    #plt.plot(x_values,y_values)
    fig, ax = plt.subplots()
    ax.plot(x_values,y_values)
    ax.scatter(x_values,y_values)
    #plt.scatter(x_values,y_values)
    plt.xticks(np.arange(0,n+1))
    plt.yticks(np.arange(0,n+1))
    ax.xaxis.tick_top()
    ax.invert_yaxis()

    mines = []
    for i in range(0,n):
        for j in range(0,n):
            if field[i,j] == -100:
                mines.append((i,j))
    if mines != []:
        mines = np.array(mines)
        mine_y = mines[:,0]
        mine_x = mines[:,1]
        for i in range(0,mines.shape[0]):
            ax.fill_between([mine_x[i],mine_x[i]+1],mine_y[i],mine_y[i]+1,color='red')
    ax.fill_between([0,1],[1,1],color='green')
    ax.fill_between([n-1,n],[n,n],[n-1,n-1],color='green')

    plt.grid()
    plt.show()


def gif(field):
    n = int(field.size**(1/2))
    list_ = np.array([(0,0)]+run(field)[1])
    y_values = list_[:,0]+0.5
    x_values = list_[:,1]+0.5
    for i in range(2,list_.shape[0]+1):
        fig, ax = plt.subplots()
        ax.plot(x_values[:i],y_values[:i])
        ax.scatter(x_values[:i],y_values[:i])
        plt.xticks(np.arange(0,n+1))
        plt.yticks(np.arange(0,n+1))
        ax.xaxis.tick_top()
        ax.invert_yaxis()
        mines = []
        for j in range(0,n):
            for k in range(0,n):
                if field[j,k] == -100:
                    mines.append((j,k))
        if mines != []:
            mines = np.array(mines)
            mine_y = mines[:,0]
            mine_x = mines[:,1]
            for l in range(0,mines.shape[0]):
                ax.fill_between([mine_x[l],mine_x[l]+1],mine_y[l],mine_y[l]+1,color='red')
        ax.fill_between([0,1],[1,1],color='green')
        ax.fill_between([n-1,n],[n,n],[n-1,n-1],color='green')
        plt.grid()
        s_no = range(0,100)
        num = s_no[i]
        print(num)
        plt.savefig(f'D:\Coding stuff\MP\source_images\plot{num}.png', dpi = 300)
    image_path = Path('source_images')
    images = list(image_path.glob('*.png'))
    image_list = []
    for file_name in images:
        image_list.append(imageio.imread(file_name))
    imageio.mimwrite('animated_from_images.gif', image_list, fps = 10)

sample = np.array([[0,0,0,0,0],[0,0,0,-100,0],[0,-100,0,-100,0],[-100,-100,0,0,0],[0,0,0,0,0]])
#gif(minefield(10,0.05))

def collect(n,p):
    no_of_mines = []
    result = []
    prob_list = []
    grid_size = []
    step_list = []
    for i in range(0,100):
        field = minefield(n,p)
        mines = []
        for j in range(0,n):
            for k in range(0,n):
                if field[j,k] == -100:
                    mines.append((j,k))
        no_of_mines.append(len(mines))
        a = run(field)
        result.append(a[0])
        prob_list.append(p)
        grid_size.append(n)
        step_list.append(len(a[1]))
    data = {"Grid size" : grid_size,"Probability" : prob_list ,"Number of mines" : no_of_mines,"Number of steps taken" : step_list,"Result" : result}
    df = pd.DataFrame(data, columns = ["Grid size","Probability","Number of mines","Number of steps taken","Result"])
    return df

# data = {"Grid size" : [],"Probability" : [] ,"Number of mines" : [],"Number of steps taken" : [] ,"Result" : []}
# df1 = pd.DataFrame(data, columns = ["Grid size","Probability","Number of mines","Number of steps taken","Result"])
# for i in range(3,24):
#     for j in np.append(np.arange(0,1,0.01),np.arange(0,0.2,0.001)):
#         df1 = df1.append(collect(i,j))
# df1.to_csv(r'D:\Coding stuff\MP\data_advance.csv', index = False)
# print(df1["Result"].value_counts())

def plot3(size):
    n = size - 3
    file = np.loadtxt('data_advance.csv',delimiter = ',',dtype = 'str')
    x_values = np.arange(0,1,0.01)
    j=1 + 30000*n
    count_list = []
    for i in range(101 + 30000*n,10002 + 30000*n,100):
        list = file[j:i,4]
        count = 0
        for k in list:
            if k == 'lost':
                count += 1
        count_list.append(count)
        j = i

    x_values_dash = np.arange(0,0.2,0.001)
    j_dash = 10001 + 30000*n
    count_list_dash = []
    for i_dash in range(10101 + 30000*n,30002 + 30000*n,100):
        list_dash = file[j_dash:i_dash,4]
        count_dash = 0
        for k_dash in list_dash:
            if k_dash == 'lost':
                count_dash += 1
        count_list_dash.append(count_dash)
        j_dash = i_dash

    x_data = np.append(x_values,x_values_dash)
    y_data = count_list + count_list_dash
    plt.scatter(x_data,y_data)
    plt.title(f"For a {size}x{size} minefield")
    plt.ylabel("No. of lost paths")
    plt.xlabel("Probability")
    plt.xticks(np.arange(0,1,0.1))
    plt.yticks(range(0,110,10))
    plt.grid()
    plt.savefig(f'D:\Coding stuff\MP\data_advance_plots\lost\ adv_n_l={size}.png', dpi = 300)
    plt.close()

# for i in range(3,24):
#     plot3(i)
# image_path = Path('data_advance_plots\lost')
# images = list(image_path.glob('*.png'))
# image_list = []
# for file_name in images:
#     image_list.append(imageio.imread(file_name))
# imageio.mimwrite('data_advance_plot_lost.gif', image_list, fps = 2)

def fit(size):
    n = size - 3
    file = np.loadtxt('data_advance.csv',delimiter = ',',dtype = 'str')
    x_values = np.arange(0,1,0.01).tolist()
    j=1 + 30000*n
    count_list = []
    for i in range(101 + 30000*n,10002 + 30000*n,100):
        listi = file[j:i,4]
        count = 0
        for k in listi:
            if k == 'success':
                count += 1
        count_list.append(count)
        j = i
    x_values_dash = np.arange(0,0.2,0.001).tolist()
    j_dash = 10001 + 30000*n
    count_list_dash = []
    for i_dash in range(10101 + 30000*n,30002 + 30000*n,100):
        list_dash = file[j_dash:i_dash,4]
        count_dash = 0
        for k_dash in list_dash:
            if k_dash == 'success':
                count_dash += 1
        count_list_dash.append(count_dash)
        j_dash = i_dash

    remove = []
    for i in range(0,len(x_values_dash)):
        for j in np.arange(0,1,0.01):
            if x_values_dash[i] == j:
                remove.append(i)
    x_values_dash_ = x_values_dash.copy()
    count_list_dash_ = count_list_dash.copy()
    for i in remove:
        x_values_dash_.remove(x_values_dash[i])
        count_list_dash_.remove(count_list_dash[i])

    x_data = x_values + x_values_dash_
    y_data = count_list + count_list_dash_

    zipped_lists = zip(x_data, y_data)
    sorted_pairs = sorted(zipped_lists)
    tuples = zip(*sorted_pairs)
    x_data, y_data = [ list(tuple) for tuple in  tuples]

    remove_indice = []
    for i in range(0,len(y_data)):
        if y_data[i] == 0:
            remove_indice.append(i)

    y_data_ = y_data.copy()
    x_data_ = x_data.copy()
    for i in remove_indice:
        y_data_.remove(y_data[i])
        x_data_.remove(x_data[i])

    y_data_ = np.array(y_data_)
    x_data_ = np.array(x_data_)


    m,c = np.polyfit(x_data_,np.log(y_data_),1)
    # plt.plot(np.array(x_data),math.e**(m*np.array(x_data) + c))
    #
    # plt.scatter(x_data,y_data)
    # plt.title(f"For a {size}x{size} minefield")
    # plt.ylabel("No. of successes")
    # plt.xlabel("Probability")
    # plt.xticks(np.arange(0,1,0.1))
    # plt.yticks(range(0,110,10))
    # plt.grid()
    # plt.show()
    return m,c

m_values = []
c_values = []
for i in range(3,24):
    m_values.append(fit(i)[0])
    c_values.append(fit(i)[1])
    print(i)
plt.scatter(range(3,24),m_values)
plt.xlabel("size of minefield (n)")
plt.ylabel("m-values")
plt.title("m(n)")
plt.grid()
plt.show()
