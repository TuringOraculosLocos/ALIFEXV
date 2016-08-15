# -*- coding: cp1252 -*-

#

# *** An Energy-Constrained Spatial Social Network Model in Python 3 ***

#

# The REDS model

# --------------

# This model generates spatially embedded social networks

# where a node's ability to build new social connections

# is limited by spatial distance (edges cannot be longer

# than a threshold "reach", R) and by cost (establishing
    
# an edge between two nodes depletes the finite "social

# energy", E, available to each of them). The cost of an

# edge between two nodes is proportional to the distance

# between the two nodes, D, and may also be influenced by

# the number of shared network neighbors that the two

# nodes possess, with the strength of this "social

# synergy" governed by the parameter S.

#

# The idea here is that:

#  1. relationships arise between nearby people (R)

#  2. relationships cost energy to maintain (E)

#  3. longer distance relationships are more costly (D)

#  4. synergy between shared social relationships may

#     reduce their cost (S).

#

# More explicitly:

#   The cost of an edge between two nodes i and j is

#       C_ij = D_ij/(1+S*K_ij)

#   D_ij is the Euclidian distance between i and j

#   D_ij must be less than R

#   K_ij is the no. network neighbors shared by i and j

#   (i.e., the no. of nodes that neighbour both i and j)

#

# The algorithm

# -------------

# Establish model parameters:

#  N (N>0): the Number of nodes

#  R (0<R<=sqrt(2)): the social Reach of all nodes

#  E (E>0): the initial social Energy budget of each node

#  S (0<=S<=1): the impact of social Synergy on edge cost 

#  p (0<=p<=1): probability that an edge in the completed network is randomly rewired 



# Place N nodes uniformly at random in the unit square

# Assign each node, i, its social Energy budget, E

# Add undirected edges sequentially as follows:

#  1) Pick a node, i, at random

#  2) Pick a node, j, at random where D_ij<R and

#     no edge ij already exists

#  3) Add the edge ij if both of the following hold:

#     - the cost of ij plus the cost of i's current edges remains less than E

#     - the cost of ij plus the cost of j's current edges remains less than E

#

# When no more edges can be added the growth terminates

#

# Then each edge is randomly rewired with probability p

#

# Acknowledgements

# ----------------

#

# This model was presented at the 14th International

# Conference on Artificial Life, 2014:

#

#   Alberto Antonioni, Seth Bullock, Marco Tomassini

#   "REDS: An Energy-Constrained Spatial Social Network

#   Model" in Hiroki Sayama, John Rieffel, Sebastian

#   Risi, René Doursat and Hod Lipson (Eds.) Artificial

#   Life 14: Proceedings of the Fourteenth International

#   Conference on the Synthesis and Simulation of Living

#   Systems, pp.368-375, MIT Press 

#

#   http://mitpress.mit.edu/sites/default/files/titles/content/alife14/ch059.html

#

# Copyright 2015- Seth Bullock

# seth.bullock@bristol.ac.uk





# ------ CODE BEGINS ------



# import all the packages that are required


import matplotlib
matplotlib.use("TkAgg")

import matplotlib.pyplot as plt

import numpy as NP

import random as RD

import pylab as PL

import networkx as NX

import math as MIT

from scipy import stats





# initialise model parameters

N = 500   # N is the Number of network nodes

R = .1    # R is the social Reach of each node  original .1

E = 0.123 # E is the social Energy of each node  inicial .123

S = 1.00      # S is the strength of social Synergy   original 1.00

P = 1.00  # p is the probability that an edge will be rewired



# calculates the Euclidian distance, D_ij, between two network nodes, i and j
    
def euclidian_dist(i,j):

    global network



    x1,y1=network.node[i]['pos'] # co-ordinates of i

    x2,y2=network.node[j]['pos'] # co-ordinates of j



    return(MIT.sqrt((x1-x2)**2+(y1-y2)**2))



# calculates the cost, C_ij, of a link between two network nodes, i and j

#   C_ij = D_ij / (1 + S*K_ij)

#   D_ij is the Euclidian distance between i and j

#   K_ij is the number of neighbours that i and j share

#   S governs the strength of social Synergy on edge cost

# (setA & setB returns whatever is in the intersection of setA and setB) 

def cost(i, j):

    global network, S, D

 

    return(D[i,j]/(1+S*(len(list(set(network.neighbors(i)) & set(network.neighbors(j)))))))



# calculate the Energy cost of a node's set of neighbours

def calc_cost(i):

    global network



    # calculate the cumulative total cost of i's current edges

    total_cost=0

    for j in network.neighbors(i):

        total_cost+=cost(i,j)



    return(total_cost)    



# update i's list of 'open' edges:

# i.e., re-establish which nodes in i's 'nearenough' list are cheap

# enough for it to afford to link to and put these in its 'open' list

# (nb any potential neighbor of i must also be able to afford the link)

def update_open_edges(i):

    global network, E

    

    network.node[i]['open']=[]                   # empty the list of i's open nodes               



    total_cost=calc_cost(i)                      # calc current energy cost for i



    for j in network.node[i]['nearenough']:

        new_cost=cost(i,j)                       # cost of potential new edge

        if total_cost+new_cost<=E:               # can i afford it?

            if calc_cost(j)+new_cost<=E:         # can j afford it?

                network.node[i]['open'].append(j)# ...add j to i's open edges



# initialises the simulation

def init():

    global time, network, N, R, E, D, S, positions, my_cmap



    #RD.seed(100) # seed the random number generator ESTO TE PERMITE RECUPERAR MUESTRAS



    time = 0 # initialise time step to zero

    

    network = NX.Graph()              # create a Graph object called network

    #NX.set_edge_attribute(network, 'weight', w)


    network.add_nodes_from(range(N))  # add N nodes to the network



    # give each node a random position in the 2-d unit square

    # give each node two initially empty node lists:

    #  'nearenough' will store a list of potential neighbor nodes

    #  'open' will store a subset of 'nearenough' that are cheap enough to add

    for i in range(N):
        #network.node[i]['pos']=[RD.uniform(0.0,0.005) for d in range(2)] #Para 5
        #network.node[i]['pos']=[RD.uniform(0.0,0.015) for d in range(2)] #Para 15
        #network.node[i]['pos']=[RD.uniform(0.0,0.05) for d in range(2)] #Para 50
        
        #network.node[i]['pos']=[RD.uniform(0.0,0.15) for d in range(2)] #Para 150
        network.node[i]['pos']=[RD.uniform(0.0,0.5) for d in range(2)] #Para 500
        #network.node[i]['pos']=[RD.uniform(0.0,1.00) for d in range(2)] #Para 1000

        #network.node[i]['pos']=[RD.random() for d in range(2)]
        # if 0 <= N < 10
        #      network.node[i]['pos']=[RD.uniform(0.0,0.005) for d in range(2)] #Para 5

        # elif 10 <= N < 30
        #     network.node[i]['pos']=[RD.uniform(0.0,0.015) for d in range(2)] #Para 15

        # elif 30 <= N < 60
        #     network.node[i]['pos']=[RD.uniform(0.0,0.035) for d in range(2)] #Para 50

        # elif 60<= N < 150
        #     network.node[i]['pos']=[RD.uniform(0.0,0.105) for d in range(2)] #Para 150

        # elif 150 <= N < 500
        #     network.node[i]['pos']=[RD.uniform(0.0,0.315) for d in range(2)] #Para 500

        # elif 500 <= N < 1500
        #     network.node[i]['pos']=[RD.uniform(0.0,1.00) for d in range(2)] #Para 1500

        network.node[i]['nearenough']=[]

        network.node[i]['open']=[]

        network.node[i]['state'] = 0


    # D is an array for storing Euclidian distances between each pair of nodes 

    D = NP.empty((N,N), dtype = float)



    # for each pair of nodes, ij, set D[i,j] and D[j,i] to be the Euclidian

    # distance between i and j. If D[i,j] is less than the social Reach

    # parameter, R, then add i to j's list of nearenough nodes and vice versa

    for i in range(N-1):

        for j in range(i+1,N):

            D[i,j]=euclidian_dist(i,j)

            D[j,i]=D[i,j]

            if D[i,j]<R:

                network.node[i]['nearenough'].append(j)

                network.node[j]['nearenough'].append(i)



    # store the locations of each node in the positions object    

    positions = NX.get_node_attributes(network,'pos')



    # for each node, check which of its potential neighbors are currently affordable

    for i in range(N):

        update_open_edges(i)



    # build a colour map: min value maps to greeny-blue, max value maps to red

    cdict = {'red': ((0.0, 0.0, 0.0),    # input zero -> 0% red 

                     (1.0, 1.0, 1.0)),   # input one -> 100% red

             'green': ((0.0, 0.75, 0.75),  # input zero -> 75% green

                     (1.0, 0.0, 0.0)),   # input one -> 0% green

             'blue': ((0.0, 1.0, 1.0),    # input zero -> 100% blue

                     (1.0, 0.0, 0.0))}   # input one -> 0% blue

    my_cmap = matplotlib.colors.LinearSegmentedColormap('my_colormap',cdict,256)



# open a plot window and set it up

def init_plot():

   global fig, ax

    

   # set up plot  

   plt.ion()                         # set "interactive" on

   fig = plt.figure(figsize=(11,11)) # 11inches square  plot inicial tamaño
   
   ax=fig.gca()

        

   # let's remove the labelled axes

   axs=plt.axes()

   axs.axes.get_yaxis().set_visible(True)     # Hide y axis

   axs.axes.get_xaxis().set_visible(True)     # Hide x axis



   # let's fix the aspect ratio even if we resize the window    

   plt.axis('scaled')

   



# perform one iteration of the model, adding a single edge

# between a pair of open nodes

#

# nb. a node, i, is said to be 'open' if there exists at least

# one legal potential new neighbor node, j, where:

#  j is not i, and j is not already a neighbour of i,

#  and D_ij<R (i.e., the Euclidian distance between i and j is

#  less than the social Reach parameter, R)

#  and, the cost of ij is affordable to both i and j 

#

# if no legal edges can be added, the function returns False,

# otherwise it returns True



def step():

    global time, network, N

    

    open_nodes=[]   # clear the list of "open" nodes



    # add i to open_nodes if there exists at least

    # one legal potential new neighbor node for i

    for i in range(N):

        if(network.node[i]['open']):

            open_nodes.append(i)

    

    if open_nodes:  # if there are any nodes in the open_nodes list...

        time += 1   # increment time step by one

    

        # indicate how many nodes currently remain open

        # print("\tTimestep: {RD.}\t {} nodes are still 'open'...".format(time,len(open_nodes)))

   

        i=RD.choice(list(open_nodes))               # choose a random open node, i

        j=RD.choice(list(network.node[i]['open']))  # choose j from i's open list



        # add an edge ij              

        network.add_edge(i, j, weight=1)



        # remove j from i's nearenough list (it's no longer a possible new neighbour)

        # remove i from j's nearenough list (it's no longer a possible new neighbour)

        network.node[i]['nearenough'].remove(j)

        network.node[j]['nearenough'].remove(i)



        # we need to update the 'open' node lists for nodes that are affected by the new edge i,j

        # these nodes are:

        #   neighbors of i and neighbors of j (including i and j themselves)

        #   any node, k, that is 'nearenough' to i or 'nearenough' to j

        to_update = network.neighbors(i) + network.neighbors(j)

        for k in range(N):

            if (k in to_update) or (i in network.node[k]['nearenough']) or (j in network.node[k]['nearenough']):

                update_open_edges(k)   

        return True

    else:

        print("There are no open nodes remaining.")

        return False

                

# analyze the network structure and write out some summary stats            

def summary():

    global network, N

    
    # set degree_sequnce to be a list of the degree of each node

    degree_sequence = list(NX.degree(network).values()) 



    # calculate the assortativity (correlation between degrees of neighbours)

    a=[]

    b=[]

    for i in range(N):

        for j in network.neighbors(i):

            a.append(network.degree(i))    

            b.append(network.degree(j))    

    r,p=stats.pearsonr(a, b)



    # print some network summary statistics

    print("max degree : %d" %max(degree_sequence))

    print("mean degree: %f" %NP.mean(degree_sequence) )

    mode=stats.mode(degree_sequence)

    print("modal degree: %d (%d nodes)" %(mode[0], mode[1]))

    print("median degree: %d" %NP.median(degree_sequence))

    print("mean degree - median degree: %f" % (NP.mean(degree_sequence)-NP.median(degree_sequence)))

    print("clustering : %f" %NX.average_clustering(network))

    print("assortativity: %f" %r)



    if(NX.is_connected(network)):

        print("mean shortest path length: %f" %NX.average_shortest_path_length(network))

    else:

        print("network is not a single component")

        largest_component=list(NX.connected_component_subgraphs(network))[0]

        size=len(largest_component)

        print("largest component has %d nodes" %size)

        print("(i.e., %f percent of the whole network)" %(size*100/N))

        print("mean shortest path length: %f" %NX.average_shortest_path_length(largest_component))



 

# draw the network

def draw():

    global network, positions, R, time, fig, my_cmap, ax



    # clear the plot    

    ax.cla()

  

    # set the title of the plot

    plt.title('REDS: Red nodes have high degree, Red edges have high cost\n'+'t = ' + str(time))



    # make a list holiding the cost of each edge    

    e = [cost(i,j) for i,j in network.edges()]

    # make a list holding the degree of each node 

    p=NX.degree_centrality(network)



    # first we draw the edges coloured by their cost (bluey-green to red)

    NX.draw_networkx_edges(network, pos = positions, edge_color=e, alpha=0.75, edge_vmax=R, edge_vmin=0.0, edge_cmap=my_cmap)



    # set the size of the plot to be just larger than the unit square IMPORTANT PART este modifica la pos
    # en hopfield   TAMAÑO DEL PLANO YA AJUSTADO POR REGLA DE 3
    if 0 <=  N < 6:
        plt.xlim(-0.01,0.007) #solo es el límite del plot

        plt.ylim(-0.01,0.007)

    elif 6 <=  N < 16:
        plt.xlim(-0.01,0.016) #solo es el límite del plot

        plt.ylim(-0.01,0.016)

    elif 16 <=  N < 51:
        plt.xlim(-0.01,0.06) #solo es el límite del plot

        plt.ylim(-0.01,0.06)

    elif 51 <=  N < 500:
        plt.xlim(-0.01,0.16) #solo es el límite del plot

        plt.ylim(-0.01,0.16)

    elif 500 <=  N < 1000:
        plt.xlim(-0.01,0.52) #solo es el límite del plot

        plt.ylim(-0.01,0.52)
    elif 1000 == N:
        plt.xlim(-0.03,1.05) #solo es el límite del plot

        plt.ylim(-0.03,1.05)





    



    # then we draw the nodes over the top of the edges coloured by their degree 

    NX.draw_networkx_nodes(network, pos = positions, node_size=80, node_color=list(p.values()), vmin=0.0, cmap=my_cmap)



    # this tiny pause allows the user to interact with the plot window

    #plt.pause(0.001)



    # let's plot the network!

    fig.canvas.draw()



    



# function to rewire each edge with probability p

#

# for each edge ij 

#   with probability p

#     remove the edge from the network

#     add i and j to a list of "nodes that belong to rewired edges"  

#     (this list may end up with multiple copies of some nodes ids)

# 

# until the list is empty:

#  pick a pair of node ids, i and j, from the list at random

#  i and j must be different nodes

#  i and j cannot be connected in the network already

#  add edge ij to the network

#  remove i and j from the list

#    

def rewire():

    global network, p, N

    

    to_rewire=[]                               # start with an empty list

    for i, j in network.edges():               # for each edge...

        if RD.random()<P:                      # ...with probability p...

            network.remove_edge(i,j)           # ...remove it...

            to_rewire.append(i)                # ...and add i and j to the list

            #to_rewire.append(j)

    for i in to_rewire:
        
        j = RD.choice(to_rewire)               # Choose one node of the list

        k = RD.choice(network.nodes())         # Choose a random node
        
        to_rewire.remove(j)                    # remove the node of the list

        network.add_edge(j,k)                  # connect the node with a the random node

        #to_rewire.remove(i)


    # while to_rewire:                           # while there are nodes in the list

    #     i=j=0                                  # set i equal to j

    #     while i==j or network.has_edge(i,j):   # while i and j are illegal

    #         i=RD.choice(to_rewire)             # pick i at random

    #         j=RD.choice(network.nodes())             # pick j at random

    #     to_rewire.remove(i)                    # remove i and j from the list

    #     #to_rewire.remove(j)

    #     network.add_edge(i,j)                  # add new edge ij to the network

          

def init2():
    global step, N, network, positions, nextNetwork, listU
    step = 0
    #For each edge, the weight of the edge, is the cost of i,j edge.
    for i,j in network.edges():
        network.edge[i][j]['weight'] = 1/cost(i,j)
    #for i, j in network.edges():  
    #        network.edge[i][j]['weight'] = cost(i,j)
    #For each node, give a random integer [1,-1] state of the node i.
    for i in range(N):
        if RD.getrandbits(1) == 1:
            network.node[i]['state'] = 1        
        else:
            network.node[i]['state'] = -1

    listU = [totalUtility()]
    #positions = NX.get_node_attributes(network,'pos')
    nextNetwork = network.copy()


#Dibuja la gráfica.
def draw2():
    PL.subplot(1, 2, 1)
    PL.cla()
    states = [network.node[i]['state'] for i in network.nodes_iter()]
    weights = [network.edge[i][j]['weight'] for [i,j] in network.edges_iter()]
    # NX.draw(network, with_labels = False, pos = positions,
    #        cmap = PL.cm.hsv, vmin = .001, vmax = .124,
    # node_color = [colorNode(s) for s in states],
    # edge_color = [colorEdge(w) for w in weights])

    NX.draw_networkx_edges(network, pos = positions, edge_color=[colorEdge(w) for w in weights], alpha=0.75, edge_vmax=R, edge_vmin=0.0, edge_cmap=my_cmap)
    NX.draw_networkx_nodes(network, pos = positions, node_size=80, node_color=[colorNode(s) for s in states], vmin=0.0, cmap=my_cmap)
    
    #Counting the number of everty type of connections
    weak = 0
    medium = 0
    strong = 0
    verystrong = 0
    for w in weights:
        if colorEdge(w) == 'green':
            weak += 1
        elif colorEdge(w) == 'blue':
            medium += 1
        elif colorEdge(w) == 'red':
            strong += 1
        elif colorEdge(w) == 'black':
            verystrong += 1
    print ('weak ', weak, 'medium', medium, 'strong', strong, 'verystrong', verystrong)


    PL.axis('image')
    PL.title('step = ' + str(step))
    PL.subplot(1, 2, 2)
    PL.cla()
    PL.plot(listU)
    PL.title('Total utility')

def step2():
    #u_i utilidad del enlace i 
    #s_i estado i
    #w_ij peso de i,j arista
    global step, N, network
    i = N
    c = 0
    azul = 0
    rojo = 0
    step += i
    #argument = 0
    #heaviside = 0
    for i in range(N):
        s_i = network.node[i]['state']
        #Counting the number of blue and red nodes
        if s_i == -1:
            azul += 1
        if s_i == 1:
            rojo += 1
        #heaviside
        argument = 0
        #print ('nodo ', s_i)
        for j in range(N):
            s_j = network.node[j]['state'] 
            if network.has_edge(i,j):
                #s_j = network.node[j]['state'] 
                w_ij = 1/cost(i,j)
                #print(w_ij)
            #w_ij = network.edge[i][j]['weight']
            #print ('estado ', s_j, 'arista ', w_ij, s_j*w_ij)
                argument += w_ij * s_j
            
                if argument < 0:
                    network.node[i]['state'] = -1
                else:
                    network.node[i]['state'] = 1

                print('rojo', rojo)
                print('azul', azul)

        #Se copia la red
        network = nextNetwork
        listU.append(totalUtility())
    #draw2()

# def CountingConnections(w):
#     weak = 0
#     medium = 0
#     strong = 0
#     verystrong = 0
#     #Weak conections
#     if 0 <= cost < 0.00175:
#         weak += 1
#     #Medium conections
#     elif 0.00175 <= w < 0.0023:
#         medium += 1
#     #Strong conections
#     elif 0.0023 <= w < 0.0035:
#         strong += 1
#     #Very strong conections
#     else:
#         verystrong += 1

#     #Statistics
#     print ('weak ', weak, 'medium', medium, 'strong', strong, 'verystrong', verystrong)


def colorNode(s):
    if s == -1:
        return 'blue'
    elif s == 1:
        return 'red'
    else:
        return 'black'

def colorEdge(w):
    #Weak conections
    if 0 <= w < 150:
        return 'green'
    #Medium conections
    elif 150 <= w < 300:
        return 'blue'
    #Strong conections
    elif 300 <= w < 450:
        return 'red'
    #Very strong conections
    else:
        return 'black'

def totalUtility():
    global network
    U = 0  
    #w_ij = network[i][j] 
    for i in range(N):
        s_i = network.node[i]['state']
        for j in range(N):
            s_j = network.node[j]['state']
            if i != j:
                if network.has_edge(i,j):
                    #s_j = network.node[j]['state']
                #w_ij = network.edge[i][j]['weight']
                    w_ij = 1/cost(i,j)
                    U += -(w_ij * s_i * s_j)
    print(U)    
    return U



# Main Model Loop

# ===============
 


growing=True        # used to indicate whether we need to keep adding more edges 



init_plot()         # initialise the plot

init()              # initialise the model



while growing:      # start the model's main loop

    growing=step()  # execute one step of the model, and update "growing"

    if time%100==0:

        draw()

    

summary()           # write out some summary stats about the network    


draw()              # draw the network


input("press [enter] to rewire")


rewire()


draw()



#input("press [enter] to rewire")


#rewire()


summary()           # write out some summary stats about the network 



#draw()              # draw the network

input("press [enter] to Hopfield")




import pycxsimulator
pycxsimulator.GUI().start(func=[init2,draw2,step2])

#init2()              # initialise the model

#step2()              # All the operations

#draw2()              #second part






# time to stop
input("finished")

