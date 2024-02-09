#@title 11. FUNCTIONS - Graph operations


import math as mt

import networkx as nx
import numpy as np

from gridworld import *


def graph_from_walls(wall_states: 'list[int]')->nx.Graph:
    '''
    Creates a graph representing the gridworld (goal state or initial state are not highlighted)
    The edges have weights corresponding to step rewards or wall rewards
    ---
    INPUT
    wall_states: wall state positions
    ---
    OUTPUT
    G: graph representing the gridworld
    '''
    G=nx.Graph()
    #add nodes
    for s in range(grid_dim**2):
        G.add_node(s,pos=(s%grid_dim,mt.floor(s/grid_dim))) #so that the coordinate (0,0) is in the middle of the gridworld
    #add "regular" edges (no wall states)
    for s1 in range(grid_dim**2):
        for s2 in [s1+1, s1+grid_dim]: #s1+1: horizontal edges, s1+grid_dim: vertical edges
            if s2 in G.nodes:
                if (s2==s1+1 and s1%grid_dim==grid_dim-1) or (s2==s1+grid_dim and s1>=grid_dim*(grid_dim-1)): #exceptions for right and top walls - here make no connection
                    continue
                else:
                    G.add_edge(s1,s2, weight=-step_reward) #we make the weights here positive so that later shortest-path-algorithms can be applied easily
    if lava:
        #for wall states, change the edge weights leading to them and away from them
        for s in wall_states:
            for (s_i,s_j) in [[s-1,s],[s,s+1],[s-grid_dim, s],[s,s+grid_dim]]: #change all edges extending from the wall state
                if (s_i,s_j) in G.edges:
                    G.edges[s_i,s_j]['weight']-=wall_reward #add the negative wall reward to these edges
    else:
        for s in wall_states:
            G.remove_node(s) #remove inaccessible wall states
    return G


def max_dist_pair(G:nx.Graph)->'tuple[int,int]':
    '''
    Compute pair of states with maximum distance between them in a given gridworld
    ---
    INPUT
    G: the graph representing the gridworld as created by graph_from_walls
    ---
    OUTPUT
    (s1,s2): The pair of states that are furthest apart form each other in the gridworld
    '''
    #find pair of nodes with maximum distance between them using dijkstra
    all_pairs=dict(nx.all_pairs_dijkstra(G))
    all_pairs_dist=[(n,all_pairs[n][0]) for n in all_pairs]
    max_dist_per_node=[(n,list(dijkdict.keys())[np.argmax(np.array(list(dijkdict.values())))],max(list(dijkdict.values()))) for n,dijkdict in all_pairs_dist]
    best_index=np.argmax([k for i,j,k in max_dist_per_node])
    s1, s2 = max_dist_per_node[best_index][0], max_dist_per_node[best_index][1]
    return s1, s2


def dead_end_goals(wall_states:'list[int]')->'list[int]':
    '''
    given a maze where lanes to walk are only a single step wide, this function calculates where possible goals can be (in dead ends)
    ---
    INPUT:
    wall_states: list of wall state positions
    ---
    OUTPUT:
    dead_ends: list of dead end states
    '''
    dead_ends=[]
    for s in range(grid_dim**2):
        #calculate neighbour "values" -> if -1 it means that there is a wall
        left_nb = -1 if s%grid_dim == 0 else s-1
        up_nb = -1 if s>=grid_dim*(grid_dim-1) else s+grid_dim
        right_nb = -1 if s%grid_dim == grid_dim-1 else s+1
        down_nb = -1 if s<grid_dim else s-grid_dim
        nbs=[right_nb,up_nb,left_nb,down_nb] #neighbours
        road_nbs=[n for n in nbs if n!=-1 and not (n in wall_states)]
        if len(road_nbs)==1 and not (s in wall_states): #has exactly one neighbour and is not a wall, then it is a possible goal
            dead_ends+=[s]
    return dead_ends


def node_dist(G: nx.Graph, s1:int, s2:int)->int:
    '''
    Compute shortest path length between two given states
    ---
    INPUT
    G: the graph representing the gridworld as created by graph_from_walls
    s1, s2: two integers representing the two states
    ---
    OUTPUT
    dist - The distance between the two nodes in steps
    '''
    #find distance between the nodes (use weight=None to count every step as distance 1)
    dist=nx.dijkstra_path_length(G,s1,s2, weight=None)
    return dist
