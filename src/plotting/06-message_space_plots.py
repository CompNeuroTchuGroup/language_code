#@title 20. PLOTS - Message space plots (PCA and t-SNE and UMAP) (Figs. 2, S1, S2, S9-12)
'''
PCA and t-SNE dimensionality reductions of the messages
'''

#general specs of the plots
world_number=len(plot_worlds)

#parameters of t-sne plots
tsne_scaling=False #if True, then all dimensions will be of equal importance
tsne_iter=1500 #number of iterations for tsne algorithm
tsne_rdkernel=1

#parameters of umap plots
umap_mindist=0.5 #minimum distance between two points (can prevent "clumping") -> (recommended: 0-1), standard 0.1
umap_rdkernel=1


#-> have to create the folder in Drive first
if not os.path.exists(file_loc+f"message plots pca/{language_code}"):
    os.mkdir(file_loc+f"message plots pca/{language_code}")
if not os.path.exists(file_loc+f"message plots tsne/{language_code}"):
    os.mkdir(file_loc+f"message plots tsne/{language_code}")
if not os.path.exists(file_loc+f"message plots umap/{language_code}"):
    os.mkdir(file_loc+f"message plots umap/{language_code}")

#plot layout details
plt.rc('axes', labelsize=32)
plt.rc('xtick', labelsize=32)
plt.rc('ytick', labelsize=32)
plt.rc('axes', titlesize=32)
markersize_all, markersize_single=35,100 #marker size for the plots involving all words and just a single world respectively
plt.rcParams["axes.prop_cycle"] = plt.cycler("color", plt.cm.tab20(np.linspace(0,world_number/20,world_number))) #color cyclwer
cmap = ColorMap2DZiegler(range_x=(-0.5-(grid_dim/2-1), 0.5+(grid_dim/2-1)), range_y=(-0.5-(grid_dim/2-1), 0.5+(grid_dim/2-1))) #2d color map for goal locations

#create a color dictionary mapping world index to color
world_color_dict={0:"grey"}
for world in range(1,grid_dim**2):
    xworld,yworld=world%grid_dim, mt.floor(world/grid_dim)
    cval=(grid_dim-1)/2
    world_color_dict[world]=cmap(xworld-cval,yworld-cval)/255


#transform messages from tensors to arrays so that PCA and t-SNE can be done on them - only keep the ones from relevant worlds
message_list=np.array([mssg.detach().cpu().numpy() for i,mssg in message_dict.items() if (label_dict[i][0] in plot_worlds and label_dict[i][2]!=student_init)])
label_list=[[i,j,k] for [i,j,k,l] in label_dict.values() if i in plot_worlds]

print(f"A total of {len(message_list)} messages are plotted")

#do the variance analysis of the message space
variance_analyzer(autoencoder,message_list, label_list, plot_worlds)

#create variance-explained plot and goal location-/initialaction-colored plots of messages
pca_plotting(message_list, label_list, language_code, "all", cmap, markersize_all)
plt.show()

#do PCA on all messages
pca=PCA(K)
message_pca=pca.fit_transform(message_list)



#FIGURE WITH COLOR BY WALL POSITION IN 2D
fig,ax=plt.subplots()
ax.xaxis.set_tick_params(width=5, length=10)
ax.yaxis.set_tick_params(width=5, length=10)
#Scatterplot of the messages, PC1 vs PC2, color by world
for world in plot_worlds:
    world_indices=[i for i,label in enumerate(label_list) if label[0]==world]
    ax.scatter(message_pca[world_indices,0],message_pca[world_indices,1],label=f"world {world}", color=world_color_dict[world], s=markersize_all)
#make some adjustments to the axes so the legend is nicely placed and doesn't overlap with data points
plt.title("Color: wall position", y=1.05)
plt.xlabel(f"first PC")
plt.ylabel(f"second PC")
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
if save_message_plots:
    plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color world.svg", format="svg")
    plt.savefig(file_loc+"message plots pca/"+language_code+f"/PCA all messages, color world.png", format="png")
plt.show()


if do_tsne_message_plots:
    #do t-SNE on all messages
    if tsne_scaling:
        message_list = StandardScaler().fit_transform(message_list) #here we standardize data -> only needed if we want all axisensions to be of equal importance
    for tsne_perplexity in [10,50]:
        tsne = TSNE(perplexity=tsne_perplexity, n_iter=tsne_iter, random_state=tsne_rdkernel, init="pca", learning_rate="auto")
        message_tsne = tsne.fit_transform(message_list)
        #Now do T-SNE plots, first by goal location and initial action
        tsne_plotting(tsne_perplexity, tsne_iter, tsne_rdkernel, message_list, label_list, language_code, "all", tsne_scaling, cmap, markersize_all)
        plt.show()

        #TSNE FIGURE WITH COLOR BY WALL POSITION IN 2D
        fig,ax=plt.subplots()
        ax.xaxis.set_tick_params(width=5, length=10)
        ax.yaxis.set_tick_params(width=5, length=10)
        for world in plot_worlds:
            world_indices=[i for i,label in enumerate(label_list) if label[0]==world]
            ax.scatter(message_tsne[world_indices,0],message_tsne[world_indices,1],label=f"world {world}", color=world_color_dict[world], s=markersize_all)
        #make some adjustments to the axes so the legend is nicely placed and doesn't overlap with data points
        plt.title(f"Color: wall position, "+r"$\pi$="+f"{tsne_perplexity}", y=1.05)
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
        if save_message_plots:
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color world, perplexity {tsne_perplexity}.svg", format="svg")
            plt.savefig(file_loc+"message plots tsne/"+language_code+f"/t-SNE all messages, color world, perplexity {tsne_perplexity}.png", format="png")
        plt.show()


if do_umap_message_plots:
    for umap_neighbors in [10,50]:
        #do UMAP on all messages
        umap_red = umap.UMAP(n_neighbors=umap_neighbors, min_dist=umap_mindist, random_state=umap_rdkernel)
        message_umap = umap_red.fit_transform(message_list)
        #Now do T-SNE plots, first by goal location and initial action
        umap_plotting(umap_neighbors, umap_mindist, umap_rdkernel, message_list, label_list, language_code, "all", cmap, markersize_all)
        plt.show()

        #UMAP FIGURE WITH COLOR BY WALL POSITION IN 2D
        fig,ax=plt.subplots()
        ax.xaxis.set_tick_params(width=5, length=10)
        ax.yaxis.set_tick_params(width=5, length=10)
        for world in plot_worlds:
            world_indices=[i for i,label in enumerate(label_list) if label[0]==world]
            ax.scatter(message_umap[world_indices,0],message_umap[world_indices,1],label=f"world {world}", color=world_color_dict[world], s=markersize_all)
        #make some adjustments to the axes so the legend is nicely placed and doesn't overlap with data points
        plt.title(f"Color: wall position, k={umap_neighbors}", y=1.05)
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
        if save_message_plots:
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color world, neighbors {umap_neighbors}.svg", format="svg")
            plt.savefig(file_loc+"message plots umap/"+language_code+f"/UMAP all messages, color world, neighbors {umap_neighbors}.png", format="png")
        plt.show()






#Finally repeat the above plots for the single grid-worlds (here only need goal location and initial action plots)
for world in plot_worlds_single:
    print(f"now looking at world {world}")
    #transform data from tensors to arrays so that PCA can be done on them
    label_list_world=[j for i,j in label_dict.items() if j[0]==world]
    message_list_world=np.array([mssg.detach().cpu().numpy() for i,mssg in message_dict.items() if label_dict[i][0]==world])
    if len(message_list_world)>0:
        #create variance-explained plot and goal location-/initial action-colored plots of messages
        pca_plotting(message_list_world, label_list_world, language_code, world, cmap, markersize_single)
        if do_tsne_message_plots:
            for tsne_perplexity_singleworld in [2,5]:
                tsne_plotting(tsne_perplexity_singleworld, tsne_iter, tsne_rdkernel,message_list_world, label_list_world, language_code, world, tsne_scaling, cmap, markersize_single)
        if do_umap_message_plots:
            for umap_neighbors_singleworld in [2,5]:
                umap_plotting(umap_neighbors_singleworld, umap_mindist, umap_rdkernel, message_list_world, label_list_world, language_code, world, cmap, markersize_single)

