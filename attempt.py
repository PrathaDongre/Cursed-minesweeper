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
            return "death",list,field_
        elif new == False:
            return "lost",list,field_
    return "success",list,field_

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
#gif(minefield(4,0.1))

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
# for i in range(3,11):
#     for j in np.arange(0,1,0.01):
#         df1 = df1.append(collect(i,j))
# df1.to_csv(r'D:\Coding stuff\MP\data.csv', index = False)
# print(df1["Result"].value_counts())

def plot3(size):
    n = size - 3
    file = np.loadtxt('data.csv',delimiter = ',',dtype = 'str')
    x_values = np.arange(0,1,0.01)
    j=1 + 10000*n
    count_list = []
    for i in range(101 + 10000*n,10002 + 10000*n,100):
        list = file[j:i,4]
        count = 0
        for k in list:
            if k == 'death':
                count += 1
        count_list.append(count)
        j = i
    plt.scatter(x_values,count_list)
    plt.title(f"For a {size}x{size} minefield")
    plt.ylabel("No. of deaths")
    plt.xlabel("Probability")
    plt.xticks(np.arange(0,1,0.1))
    plt.yticks(range(0,110,10))
    plt.grid()
    plt.savefig(f'D:\Coding stuff\MP\data_plots\deaths\ n_d={size}.png', dpi = 300)
    plt.close()

# for i in range(3,11):
#     plot3(i)
# image_path = Path('data_plots\deaths')
# images = list(image_path.glob('*.png'))
# image_list = []
# for file_name in images:
#     image_list.append(imageio.imread(file_name))
# imageio.mimwrite('data_plot_deaths.gif', image_list, fps = 2)

def fit(size):
    n = size - 3
    file = np.loadtxt('data.csv',delimiter = ',',dtype = 'str')
    x_values = np.arange(0,1,0.01).tolist()
    j=1 + 10000*n
    count_list = []
    for i in range(101 + 10000*n,10002 + 10000*n,100):
        list = file[j:i,4]
        count = 0
        for k in list:
            if k == 'success':
                count += 1
        count_list.append(count)
        j = i

    remove_indice = []
    for i in range(0,len(count_list)):
        if count_list[i] == 0:
            remove_indice.append(i)

    count_list_ = count_list.copy()
    x_values_ = x_values.copy()
    for i in remove_indice:
        count_list_.remove(count_list[i])
        x_values_.remove(x_values[i])

    count_list_ = np.array(count_list_)
    x_values_ = np.array(x_values_)

    m,c = np.polyfit(x_values_,np.log(count_list_),1)
    plt.plot(np.array(x_values),math.e**(m*np.array(x_values) + c))

    #plt.scatter(x_values,count_list)
    plt.title(f"For a {size}x{size} minefield")
    plt.ylabel("No. of successes")
    plt.xlabel("Probability")
    plt.xticks(np.arange(0,1,0.1))
    plt.yticks(range(0,110,10))
    plt.grid()
    #return m,c

# m_values = []
# c_values = []
# for i in range(3,11):
#     m_values.append(fit(i)[0])
#     c_values.append(fit(i)[1])
#     print(i)
# plt.scatter(range(3,11),c_values)
# plt.xlabel("Minefield Size (n)")
# plt.ylabel("c-values")
# plt.title("c(n)")
# plt.grid()
# plt.show()

for i in range(3,11):
    fit(i)
plt.title("Best Fit Curves")
plt.legend(["n = 3","n = 4","n = 5","n = 6","n = 7","n = 8","n = 9","n = 10"])
plt.grid()
plt.show()
