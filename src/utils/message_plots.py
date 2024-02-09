#@title 15. FUNCTIONS - For message plots


import math as mt

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import umap.umap_ as umap
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.preprocessing import StandardScaler

from dqn_optimizers import *


def variance_analyzer(autoencoder: nn.Module,message_list: 'list[np.array]', label_list: 'list[tuple[int,int,int]]', plot_worlds)-> 'tuple[float, float, float,float]':
    '''
    find out the variance contributions of goal location and world identity (and maybe initial action or other quantities)
    we compare within-group variance to between-group variance - use ANOVA (ANALYSIS OF VARIANCE)
    1.between-group-variance computes variance of the mean message values of groups
    2.within-group-variance computes variance of messages within groups and sums them up
    Take care of the case where the groups might have different standard deviations but the same mean, then between-group variance is low despite the groups
    being very different
    ---
    INPUT
    autoencoder - the language proxy network generating Q-matrices from messages
    message_list - all messages we are considering
    label_list - the task labels corresponding to the messages we are considering
    plot_worlds - list of all plotted worlds (with indices)
    ---
    OUTPUT
    relvart_worlds - relative variance (between/between+within) when considering "world groups"
    relvart_goals - relative variance (between/between+within) when considering "goal groups"
    relvart_actions - relative variance (between/between+within) when considering "action groups"
    discarded_pctg - percentage of discarded messages for action variance (those where probability of walking into wall at beginning is above 20%)
    '''

    #sort the messages into groups (messages within a group are tasks from the same gridworld/with identical goal locations)
    #in both cases, we exclude world 0 from the analysis -> goal 0 was already excluded in Q-matrix generation
    world_groups_messages=[[message_list[i] for i,label in enumerate(label_list) if label[0]==j if label[2]!=student_init] for j in plot_worlds if j!=0]
    goal_groups_messages=[[message_list[i] for i,label in enumerate(label_list) if label[2]==j and label[0]!=0] for j in range(grid_dim**2) if j!=student_init]

    #number of groups
    K_world=len(world_groups_messages)
    K_goal=len(goal_groups_messages)
    #number of total messages
    N_world=sum([len(i) for i in world_groups_messages])
    N_goal=sum([len(i) for i in goal_groups_messages])

    print(f"We include {N_world} messages in the {K_world} groups in the variance analysis with world groupings")
    print(f"We include {N_goal} messages in the {K_goal} groups in the variance analysis with goal groupings")

    #remove empty groups
    world_groups_messages=[group for group in world_groups_messages if len(group)!=0]
    goal_groups_messages=[group for group in goal_groups_messages if len(group)!=0]

    #calculate group means
    world_message_means=[[len(group),np.mean(group, axis=0)] for group in world_groups_messages]
    goal_message_means=[[len(group),np.mean(group, axis=0)] for group in goal_groups_messages]

    #calculate overall mean
    world_mean_message=np.sum([i*j for i,j in world_message_means], axis=0)/N_world
    goal_mean_message=np.sum([i*j for i,j in world_message_means], axis=0)/N_goal


    #between group variation
    var_between_worlds=np.sum([i*(np.linalg.norm(j-world_mean_message,2)**2) for i,j in world_message_means])
    var_between_goals=np.sum([i*(np.linalg.norm(j-goal_mean_message,2)**2) for i,j in goal_message_means])

    #within group variation
    var_within_worlds=np.sum([np.linalg.norm(j-world_message_means[i][1],2)**2 for i,group in enumerate(world_groups_messages) for j in group])
    var_within_goals=np.sum([np.linalg.norm(j-goal_message_means[i][1],2)**2 for i,group in enumerate(goal_groups_messages) for j in group])

    #calculate relative variations (beta)-> ratio of between-group variation to the sum of between-group and within-group variation
    beta_world=round(var_between_worlds/(var_within_worlds+var_between_worlds),3)
    beta_goal=round(var_between_goals/(var_within_goals+var_between_goals),3)

    #calculate F-values:
    #first mean squares between and within groups
    MSb_world, MSb_goal=var_between_worlds/(K_world-1), var_between_goals/(K_goal-1)
    MSw_world, MSw_goal=var_within_worlds/(N_world-K_world), var_within_goals/(N_goal-K_goal)
    #finally get F-value
    F_world, F_goal=MSb_world/MSw_world, MSb_goal/MSw_goal

    print(f"World groups: var. within {round(var_within_worlds,2)}, var. between {round(var_between_worlds,2)}, beta {beta_world}, F {round(F_world,2)}")
    print(f"Goal groups: var. within {round(var_within_goals,2)}, var. between {round(var_between_goals,2)}, beta {beta_goal}, F {round(F_goal,2)}")
    print("")





def pca_plotting(autoencoder, message_list: 'list[np.array]', label_list: 'dict[int,tuple[int, int, int]]', language_code: str, world_add_on: str, cmap, markersize,
               ):
    '''
    Does a PCA on a list of messages and creates two plots
    One barplot for the variance explained by PC
    One scatterplot for the messages, where PC1 vs PC2 is plotted and the messages are colored according to goal location
    ---
    INPUT
    message_list - list of messages to consider/plot
    label_list - list of tuples (wall_index, initial_state, goal_state) corresponding to the messages (wall index from wall state dict)
    language_add_on - Marks the language from which we are analyzing the messages -> the corresponding folder where we store the figures
    world_add_on - Marks in the plot title if we are considering all worlds or only one specific world
    cmap - 2d colormap
    ---
    OUTPUT
    The plot.
    '''


    pca=PCA(K)
    pca.fit(message_list)
    message_pca=pca.fit_transform(message_list)




    #FIGURE WITH BARS SHOWING VARIANCE EXPLAINED BY PC
    cmap2=plt.get_cmap("bwr")
    axt = plt.gca()
    axt.tick_params(width=5, length=10)
    #Barplot of variance explained for all messages
    var_arr=np.round_(pca.explained_variance_ratio_*100,2)
    plt.bar(range(min(K+1,11))[1:],var_arr, color="gray") #for each message dimension have one PC
    plt.xlabel(f"PC index")
    plt.ylabel(r"Variance" "\n" r"explained (%)")
    plt.xticks(range(min(K+1,11))[1:])
    plt.yticks([0,25,50,75,100])
    plt.ylim(0,100)
    print(f"explained variances by the first {min(K,10)} PCs are {var_arr[:min(K,10)]}")

    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    if save_message_plots:
        if world_add_on=="all":
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/Variance explained all messages.svg",  format="svg")
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/Variance explained all messages.png",  format="png")
        else:
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/Variance explained messages world(s) {world_add_on}.svg", format="svg")
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/Variance explained messages world(s) {world_add_on}.png", format="png")
    plt.show()




    #FIGURE WITH COLOR BY GOAL LOCATION IN 2D
    fig,ax = plt.subplots()
    ax.xaxis.set_tick_params(width=5, length=10)
    ax.yaxis.set_tick_params(width=5, length=10)
    for i, label in enumerate(label_list):
        goal=label[2]
        #transform to goal coordinates
        cval=(grid_dim-1)/2 #center (0,0) is in the middle of the grid
        goal_coords=goal%grid_dim-cval, mt.floor(goal/grid_dim)-cval
        #plot datapoint
        ax.scatter(message_pca[i,0],message_pca[i,1],color=cmap(goal_coords[0],goal_coords[1])/255, s=markersize)

    #make some adjustments to the axes so the legend is nicely placed and doesn't overlap with data points
    ax.set_xlabel(f"first PC")
    xdatamin,xdatamax=min(message_pca[:,0]), max(message_pca[:,0])
    ydatamin,ydatamax=min(message_pca[:,1]), max(message_pca[:,1])
    xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
    ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
    xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

    #typical case of PCA
    if xdatamax-xdatamin > ydatamax-ydatamin:
        ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
        ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
    #for t-sne and umap have to take other scenario into account
    else:
        ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
        ax.set_ylim(ymin, ymax)

    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    plt.title("Color: goal location", y=1.05)
    if save_message_plots:
        if world_add_on=="all":
            ax.tick_params(left=False, labelleft=False) #remove ticks and labels from y axis
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color goal.svg", format="svg")
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color goal.png", format="png")
        else:
            ax.set_ylabel(f"second PC")
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA messages, color goal, world(s) {world_add_on}.svg", format="svg")
            plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA messages, color goal, world(s) {world_add_on}.png", format="png")
    plt.show()




    if not language_code.__contains__("nostudent"):
        #FIGURE WITH COLOR BY INITIAL ACTION IN 2D
        fig,ax = plt.subplots()
        ax.xaxis.set_tick_params(width=5, length=10)
        ax.yaxis.set_tick_params(width=5, length=10)
        softy=nn.Softmax(dim=0)
        for i,message in enumerate(message_list):
            state_tensors=get_state_tensors(1)
            Q=autoencoder.student(torch.tensor([message]), state_tensors)[0]
            probas=softy(Q[:,0%grid_dim,mt.floor(0/grid_dim)]).detach().cpu().numpy()
            extr=0.5+grid_dim/2-1
            comb2d=probas[0]*np.array([extr,extr])+probas[1]*np.array([extr,-extr])+probas[2]*np.array([-extr,-extr])+probas[3]*np.array([-extr,extr])
            ax.scatter(message_pca[i,0],message_pca[i,1],color=cmap(comb2d[0],comb2d[1])/255, s=markersize)

        plt.xlabel(f"first PC")
        #plt.ylabel(f"second PC")
        plt.tick_params(left=False, labelleft=False) #remove ticks and labels from y axis
        xdatamin,xdatamax=min(message_pca[:,0]), max(message_pca[:,0])
        ydatamin,ydatamax=min(message_pca[:,1]), max(message_pca[:,1])
        xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
        ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
        xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

        #typical case of PCA
        if xdatamax-xdatamin > ydatamax-ydatamin:
            ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
            ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
        #for t-sne and umap have to take other scenario into account
        else:
            ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
            ax.set_ylim(ymin, ymax)
        plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
        plt.title(f"Color: first student action", y=1.05)
        if save_message_plots:
            if world_add_on=="all":
                plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color action.svg",  format="svg")
                plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color action.png",  format="png")
            else:
                plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA messages, color action, world(s) {world_add_on}.svg",  format="svg")
                plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA messages, color action, world(s) {world_add_on}.png",  format="png")
        plt.show()





def tsne_plotting(autoencoder, perplexity: int, n_iter: int, rdkernel: int, message_list: 'list[np.array]', label_list: 'dict[int,tuple[int, int, int]]', language_code: str,
                  world_add_on: str, scaling: bool, cmap, markersize, ):
    '''
    Does a t-SNE on a list of messages and creates a plot, namely
    a 2D scatterplot for the transformed messages, which are colored according to goal location
    ---
    INPUT
    perplexity - something like number of neighbours to consider for the tsne-algorithm (need integer here instead of float or some weird format error in plotting will occur)
    n_iter - number of iterations for tsne algorithm
    rdkernel - random seed
    message_list - list of messages to consider/plot
    label_list - list of tuples (wall_index, initial_state, goal_state) corresponding to the messages (wall index from wall state dict)
    language_add_on - Marks the language from which we are analyzing the messages -> the corresponding folder where we store the figures
    world_add_on - Marks in the plot title if we are considering all worlds or only one specific world
    scaling - if True, scaling is performed before tsne application (only recommended if all dimensions should be treated with same importance)
    cmap - 2d colormap
    ---
    OUTPUT
    The plot.
    '''

    tsne = TSNE(perplexity=perplexity, n_iter=n_iter, random_state=rdkernel, init="pca", learning_rate="auto")
    if scaling:
        message_list=StandardScaler().fit_transform(message_list)
    message_tsne = tsne.fit_transform(np.array(message_list))




    #FIGURE WITH COLOR BY GOAL LOCATION IN 2D
    fig,ax=plt.subplots()
    ax.xaxis.set_tick_params(width=5, length=10)
    ax.yaxis.set_tick_params(width=5, length=10)
    for i, label in enumerate(label_list):
        goal=label[2]
        #transform to goal coordinates
        cval=(grid_dim-1)/2 #center (0,0) is in the middle of the grid
        goal_coords=goal%grid_dim-cval, mt.floor(goal/grid_dim)-cval
        #plot datapoint
        ax.scatter(message_tsne[i,0],message_tsne[i,1],color=cmap(goal_coords[0],goal_coords[1])/255, s=markersize)

    xdatamin,xdatamax=min(message_tsne[:,0]), max(message_tsne[:,0])
    ydatamin,ydatamax=min(message_tsne[:,1]), max(message_tsne[:,1])
    xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
    ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
    xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

    #typical case of PCA
    if xdatamax-xdatamin > ydatamax-ydatamin:
        ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
        ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
    #for t-sne and umap have to take other scenario into account
    else:
        ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
        ax.set_ylim(ymin, ymax)
    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    plt.title(f"Color: goal location, "+r"$\pi$="+f"{perplexity}", y=1.05)
    if save_message_plots:
        if world_add_on=="all":
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color goal, perplexity {perplexity}.svg", format="svg")
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color goal, perplexity {perplexity}.png", format="png")
        else:
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE messages world(s) {world_add_on}, color goal, perplexity {perplexity}.svg", format="svg")
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE messages world(s) {world_add_on}, color goal, perplexity {perplexity}.png", format="png")
    plt.show()



    if not language_code.__contains__("nostudent"):
        fig,ax=plt.subplots()
        ax.xaxis.set_tick_params(width=5, length=10)
        ax.yaxis.set_tick_params(width=5, length=10)
        #FIGURE WITH COLOR BY INITIAL ACTION IN 2D
        softy=nn.Softmax(dim=0)
        for i,message in enumerate(message_list):
            state_tensors=get_state_tensors(1)
            Q=autoencoder.student(torch.tensor([message]), state_tensors)[0]
            probas=softy(Q[:,student_init%grid_dim,mt.floor(student_init/grid_dim)]).detach().cpu().numpy()
            extr=0.5+grid_dim/2-1
            #combine the four probabilities into a 2-dimensional vector using the four corners of a square with four different colours
            comb2d=probas[0]*np.array([extr,extr])+probas[1]*np.array([extr,-extr])+probas[2]*np.array([-extr,-extr])+probas[3]*np.array([-extr,extr])
            ax.scatter(message_tsne[i,0],message_tsne[i,1],color=cmap(comb2d[0],comb2d[1])/255, s=markersize)

        xdatamin,xdatamax=min(message_tsne[:,0]), max(message_tsne[:,0])
        ydatamin,ydatamax=min(message_tsne[:,1]), max(message_tsne[:,1])
        xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
        ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
        xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

        #typical case of PCA
        if xdatamax-xdatamin > ydatamax-ydatamin:
            ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
            ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
        #for t-sne and umap have to take other scenario into account
        else:
            ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
            ax.set_ylim(ymin, ymax)
        plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
        plt.title(f"Color: first student action, "+r"$\pi$="+f"{perplexity}", y=1.05)
        if save_message_plots:
            if world_add_on=="all":
                plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color action, perplexity {perplexity}.svg", format="svg")
                plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color action, perplexity {perplexity}.png", format="png")
            else:
                plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE messages world(s) {world_add_on}, color action, perplexity {perplexity}.svg", format="svg")
                plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE messages world(s) {world_add_on}, color action, perplexity {perplexity}.png", format="png")
        plt.show()








def umap_plotting(autoencoder, neighbors: int, min_dist: float, rdkernel: int, message_list: 'list[np.array]', label_list: 'dict[int,tuple[int, int, int]]', language_code: str,
                  world_add_on: str, cmap, markersize):
    '''
    Does a Umap on a list of messages and creates a plot, namely
    a 2D scatterplot for the transformed messages, which are colored according to goal location
    ---
    INPUT
    neighbors - parameter of UMAP that controls tradeoff between local and global properties
    min_dist - minimum distance between two points in the final embedding (prevent clumping if >0)
    rdkernel - random seed
    message_list - list of messages to consider/plot
    label_list - list of tuples (wall_index, initial_state, goal_state) corresponding to the messages (wall index from wall state dict)
    language_add_on - Marks the language from which we are analyzing the messages -> the corresponding folder where we store the figures
    world_add_on - Marks in the plot title if we are considering all worlds or only one specific world
    cmap - 2d colormap
    ---
    OUTPUT
    The plot.
    '''

    umap_red = umap.UMAP(n_neighbors=neighbors, min_dist=min_dist, random_state=rdkernel)
    message_umap = umap_red.fit_transform(np.array(message_list))


    #FIGURE WITH COLOR BY GOAL LOCATION IN 2D
    fig,ax=plt.subplots()
    ax.xaxis.set_tick_params(width=5, length=10)
    ax.yaxis.set_tick_params(width=5, length=10)
    for i, label in enumerate(label_list):
        goal=label[2]
        #transform to goal coordinates
        cval=(grid_dim-1)/2 #center (0,0) is in the middle of the grid
        goal_coords=goal%grid_dim-cval, mt.floor(goal/grid_dim)-cval
        #plot datapoint
        ax.scatter(message_umap[i,0],message_umap[i,1],color=cmap(goal_coords[0],goal_coords[1])/255, s=markersize)

    xdatamin,xdatamax=min(message_umap[:,0]), max(message_umap[:,0])
    ydatamin,ydatamax=min(message_umap[:,1]), max(message_umap[:,1])
    xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
    ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
    xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

    #typical case of PCA
    if xdatamax-xdatamin > ydatamax-ydatamin:
        ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
        ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
    #for t-sne and umap have to take other scenario into account
    else:
        ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
        ax.set_ylim(ymin, ymax)
    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    plt.title(f"Color: goal location, k={neighbors}", y=1.05)
    if save_message_plots:
        if world_add_on=="all":
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color goal, neighbors {neighbors}.svg", format="svg")
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color goal, neighbors {neighbors}.png", format="png")
        else:
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP messages world(s) {world_add_on}, color goal, neighbors {neighbors}.svg", format="svg")
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP messages world(s) {world_add_on}, color goal, neighbors {neighbors}.png", format="png")
    plt.show()




    if not language_code.__contains__("nostudent"):
        fig,ax=plt.subplots()
        ax.xaxis.set_tick_params(width=5, length=10)
        ax.yaxis.set_tick_params(width=5, length=10)
        #FIGURE WITH COLOR BY INITIAL ACTION IN 2D
        softy=nn.Softmax(dim=0)
        for i,message in enumerate(message_list):
            state_tensors=get_state_tensors(1)
            Q=autoencoder.student(torch.tensor([message]), state_tensors)[0]
            probas=softy(Q[:,student_init%grid_dim,mt.floor(student_init/grid_dim)]).detach().cpu().numpy()
            extr=0.5+grid_dim/2-1
            #combine the four probabilities into a 2-dimensional vector using the four corners of a square with four different colours
            comb2d=probas[0]*np.array([extr,extr])+probas[1]*np.array([extr,-extr])+probas[2]*np.array([-extr,-extr])+probas[3]*np.array([-extr,extr])
            ax.scatter(message_umap[i,0],message_umap[i,1],color=cmap(comb2d[0],comb2d[1])/255, s=markersize)

        xdatamin,xdatamax=min(message_umap[:,0]), max(message_umap[:,0])
        ydatamin,ydatamax=min(message_umap[:,1]), max(message_umap[:,1])
        ymean=(ydatamax+ydatamin)/2
        xmin, xmax = xdatamin - 0.07*(xdatamax-xdatamin), xdatamax + 0.07*(xdatamax-xdatamin) #x axis limits a bit larger than data range to not cut points in the middle
        ymin, ymax = ydatamin - 0.07*(ydatamax-ydatamin), ydatamax + 0.07*(ydatamax-ydatamin)
        xmean, ymean=(xmax+xmin)/2, (ymax+ymin)/2

        #typical case of PCA
        if xdatamax-xdatamin > ydatamax-ydatamin:
            ax.set_xlim(xmin,xmax+(xmax-xmin)/2)
            ax.set_ylim(ymean-1/2*(xmax-xmin), ymean+1/2*(xmax-xmin)) #manually set equal aspect ratio, it was the only way that worked..
        #for t-sne and umap have to take other scenario into account
        else:
            ax.set_xlim(xmean-1/2*(ymax-ymin), xmean+1*(ymax-ymin))
            ax.set_ylim(ymin, ymax)
        plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
        plt.title(f"Color: first student action, k={neighbors}", y=1.05)
        if save_message_plots:
            if world_add_on=="all":
                plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color action, neighbors {neighbors}.svg", format="svg")
                plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color action, neighbors {neighbors}.png", format="png")
            else:
                plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP messages world(s) {world_add_on}, color action, neighbors {neighbors}.svg", format="svg")
                plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP messages world(s) {world_add_on}, color action, neighbors {neighbors}.png", format="png")
        plt.show()
