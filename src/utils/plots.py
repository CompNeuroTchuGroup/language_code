#@title 10. FUNCTIONS - Plots of Q-matrices and probability matrices (from Q matrices)

import math as mt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
from matplotlib.font_manager import FontProperties


def q_probabilities_plotter(Q: torch.tensor, init_state:int, goal_state:int, wall_states:'list[int]', steps:int, title: str):
    '''
    Plot the probabilites of state occupancy after a certain number of steps in a grid plot
    The initial state is marked with green text color, the final state with gold text color
    ---
    INPUT
    Q: the q-matrix to be used for probability calculation
    init_state, goal_state, wall_states: properties of the gridworld
    steps: number of steps for the agent after which the probabilites are calculated
    title: title for the plot
    ---
    OUTPUT
    Nothing except the plot
    '''

    cmap=plt.get_cmap("binary")
    softy=nn.Softmax(dim=0)
    #first build the network architecture
    env = SquareGridworld(init_state,goal_state,wall_states, lava)
    outcomes = env.get_outcomes()
    #dictionary to retrieve next state and reward given current (s,a)
    next_states_dict={s:[outcomes[s,a][0] for a in range(n_actions)]  for s in range(grid_dim**2)}
    matrix_big=torch.zeros(size=(grid_dim**2,grid_dim**2), device=device) #create a big transition matrix
    #get action probabilities as softmax of the Q-values
    action_probas=softy(Q)
    for s in range(grid_dim**2):
        if s!=goal_state:
            for i,ns in enumerate(next_states_dict[s]):
                matrix_big[ns,s]+=action_probas[i,mt.floor(s/grid_dim),s%grid_dim] #need to have the "+=" here because there can be multiple identical transitions, e.g. for corner states
        else:
            matrix_big[s,s]=1
    #initialize probabilities
    probas=torch.zeros(grid_dim**2, device=device)
    probas[init_state]=1
    #Apply the transition matrix to the initial probability distribution for each allowed step to get the final probability distribution
    for _ in range(steps):
        probas=matrix_big@probas

    xydict={s:(grid_dim-1-mt.floor(s/grid_dim), s%grid_dim) for s in range(grid_dim**2)}
    #Create 2D lists for state probabilites and state colors
    state_probas=np.zeros(shape=(grid_dim,grid_dim))
    state_colors=np.zeros(shape=(grid_dim,grid_dim))
    state_probas=[list(arr) for arr in state_colors]
    state_colors=[list(arr) for arr in state_colors]
    #Insert probabilites and colors at the correct positions (so that state 0 is bottom left and state grid_dim**2-1 is top right)
    for s in range(grid_dim**2):
        x,y=xydict[s]
        if s in wall_states:
            state_probas[x][y]=f"0%"
            state_colors[x][y]=cmap(1-probas[s].item())
        else:
            state_probas[x][y]=f"{round(100*probas[s].item(),2)}%"
            state_colors[x][y]=cmap(0.75*(1-probas[s].item()))


    #Now plot as 2D plot with probabilities and state numbers
    fig, ax = plt.subplots()
    # hide axes
    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')
    table=ax.table(cellText=state_probas, cellColours=state_colors, loc='center', colWidths=[0.2]*grid_dim, cellLoc='center')
    #some tweaking of font and text color (initial state green, goal state gold)
    table._cells[xydict[init_state]]._text.set_color('darkgreen')
    table._cells[xydict[goal_state]]._text.set_color('goldenrod')
    table.scale(0.72,4)
    for (row, col), cell in table.get_celld().items():
        cell.set_text_props(fontproperties=FontProperties(size=35))
    fig.tight_layout()
    plt.title(title)
    plt.show()



def q_arrows_plotter(Q:torch.tensor, init_state: int, goal_state:int, wall_states:'list[int]', bad_trust_states: 'list[int]',
                     good_trust_states: 'list[int]', title: str, save_loc: str, save: bool):
    '''
    Plot the probabilites of state transitions as arrows in a grid plot
    The initial state is marked with green color, the final state with gold color
    ---
    INPUT
    Q: the q-matrix to be used for probability calculation
    init_state, goal_state, wall_states:  properties of the gridworld
    bad_trust_states/good_trust_states: states in which the trust was bad/good
    title: title for the plot
    save_loc: saving location for the plot.
    save: True if we should save the figure, False otherwise
    ---
    OUTPUT
    Nothing except the plot
    '''
    plot_dim=grid_dim+2 #outside walls on each side mean the dimension is increased by 2
    #change wall states to incorporate outside states
    outer_wall_states=[i for i in range(plot_dim**2) if i<plot_dim or i%plot_dim in [plot_dim-1,0] or i>plot_dim*(plot_dim-1)]
    outer_bt_states, outer_gt_states= [],[]
    for w in wall_states:
        w1,w2=w%grid_dim, mt.floor(w/grid_dim)
        outer_wall_states+=[plot_dim*w2+w1+plot_dim+1]
    wall_states=outer_wall_states
    i1,i2=init_state%grid_dim, mt.floor(init_state/grid_dim)
    init_state=plot_dim*i2+i1+plot_dim+1
    g1,g2=goal_state%grid_dim, mt.floor(goal_state/grid_dim)
    goal_state=plot_dim*g2+g1+plot_dim+1
    for b in bad_trust_states:
        b1,b2=b%grid_dim, mt.floor(b/grid_dim)
        outer_bt_states+=[plot_dim*b2+b1+plot_dim+1]
    bad_trust_states=outer_bt_states
    for g in good_trust_states:
        g1,g2=g%grid_dim, mt.floor(g/grid_dim)
        outer_gt_states+=[plot_dim*g2+g1+plot_dim+1]
    good_trust_states=outer_gt_states


    cmap=plt.get_cmap("binary")
    softy=nn.Softmax(dim=0)
    #get action probabilities as softmax of the Q-values
    action_probas=softy(Q)

    xydict={s:(plot_dim-1-mt.floor(s/plot_dim), s%plot_dim) for s in range(plot_dim**2)} #+1 to include outside walls
    #Create 2D lists for state probabilites and state colors
    state_colors=np.zeros(shape=(plot_dim,plot_dim))
    state_colors=[list(arr) for arr in state_colors]
    for s in range(plot_dim**2):
        x,y=xydict[s]
        if s in wall_states:
            state_colors[x][y]=(122/255, 176/255, 207/255,1)
        elif s in bad_trust_states:
            state_colors[x][y]=(255/255, 170/255, 170/255,1)
        elif s in good_trust_states:
            state_colors[x][y]=(144/255, 238/255, 144/255,1)
        else:
            state_colors[x][y]=cmap(0.)

        '''
        elif s==init_state:
            state_colors[x][y]=(239/255, 138/255, 98/255,1)
        elif s==goal_state:
            state_colors[x][y]=(95/255, 182/255, 119/255,1)
        '''

    #Now plot as 2D plot with probabilities and state numbers
    fig, ax = plt.subplots(figsize=(8,8))
    # hide axes
    fig.patch.set_visible(True)
    ax.axis('off')
    ax.axis('tight')
    table=ax.table(cellColours=state_colors, loc='center', colWidths=[0.2]*plot_dim, cellLoc='center')
    table.scale(1,7)
    cell_size_x, cell_size_y=[0.0221,0.0212]  #have to somehow find out the cell size to plot proper arrows
    for s in range(plot_dim**2):
        if s in wall_states or s==goal_state:
            continue
        x,y=cell_size_x*(s%plot_dim-plot_dim/2+0.5), cell_size_y*(mt.floor(s/plot_dim)-plot_dim/2+0.5) #coordinates on the plot of our state
        #transform back the state index
        s_original=s-plot_dim-1
        if s>=2*plot_dim:
            s_original-=2*mt.floor(s/plot_dim-1)
        indx,indy=mt.floor(s_original/grid_dim),s_original%grid_dim
        probas=np.round_(action_probas[:,indx,indy].detach().cpu().numpy(),2)
        maxdir=np.argmax(probas)
        #all four possible directions
        wbase=1.8e-3
        hb=2e-3
        hb2=3e-3
        if maxdir==0:
            plt.arrow(x+cell_size_x/4,y+cell_size_y/8,cell_size_x/2,0, width=wbase*mt.sqrt(probas[0]),length_includes_head=True, head_width=hb+hb2*mt.sqrt(probas[0]), head_length=hb+hb2*mt.sqrt(probas[0]), color="firebrick") #right
            #plt.text(x+cell_size_x/4,y+cell_size_y/6, f"{round(100*probas[0])}", horizontalalignment="left", verticalalignment="bottom")
        elif maxdir==2:
            plt.arrow(x-cell_size_x/4,y-cell_size_y/8,-cell_size_x/2,0, width=wbase*mt.sqrt(probas[2]),length_includes_head=True, head_width=hb+hb2*mt.sqrt(probas[2]), head_length=hb+hb2*mt.sqrt(probas[2]),color="firebrick") #left
            #plt.text(x-cell_size_x/4,y-cell_size_y/12, f"{round(100*probas[2])}", horizontalalignment="right", verticalalignment="bottom")
        elif maxdir==1:
            plt.arrow(x+cell_size_x/8,y+cell_size_y/4,0,cell_size_y/2, width=wbase*mt.sqrt(probas[1]),length_includes_head=True, head_width=hb+hb2*mt.sqrt(probas[1]), head_length=hb+hb2*mt.sqrt(probas[1]),color="firebrick") #up
            #plt.text(x+cell_size_x/6,y+3*cell_size_y/4, f"{round(100*probas[1])}", horizontalalignment="left", verticalalignment="top")
        else:
            plt.arrow(x-cell_size_x/8,y-cell_size_y/4,0,-cell_size_y/2, width=wbase*mt.sqrt(probas[3]),length_includes_head=True, head_width=hb+hb2*mt.sqrt(probas[3]), head_length=hb+hb2*mt.sqrt(probas[3]),color="firebrick") #down
            #plt.text(x-cell_size_x/6,y-3*cell_size_y/4, f"{round(100*probas[3])}", horizontalalignment="right", verticalalignment="bottom")
    plt.title(title)
    if save:
        plt.savefig(save_loc, format="svg")
        plt.figure()
    if not save:
        plt.show()


