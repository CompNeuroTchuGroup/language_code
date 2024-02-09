#@title 21. PLOTS - Autoencoder loss plots (Figs. 4a-d, S6)

#plotting details
plt.rc('axes', labelsize=36)
plt.rc('xtick', labelsize=36)
plt.rc('ytick', labelsize=36)
plt.rc('axes', titlesize=36)
plt.rcParams['svg.fonttype']='none' #to make later editing of figures in inkscape easier
cmap=plt.get_cmap("bwr")



#load losses from case without student feedback
losses1=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_losses1_factor1.txt")
rec_losses1=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_rec_losses1_factor1.txt")
spar_losses1=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_spar_losses1_factor1.txt")
#load losses from case with student feedback
rec_losses2=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_rec_losses2_zeta{zeta_lossplot}_factor1.txt")
losses2=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_losses2_zeta{zeta_lossplot}_factor1.txt")
spar_losses2=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_spar_losses2_zeta{zeta_lossplot}_factor1.txt")
goal_losses2=np.loadtxt(file_loc+f"autoencoder/losses/nonlinear_goal_losses2_zeta{zeta_lossplot}_factor1.txt")



#Plot reconstruction losses with and without student feedback
fig,ax = plt.subplots()
ax.xaxis.set_tick_params(width=5, length=10)
ax.yaxis.set_tick_params(width=5, length=10)

ax.plot(range(epskip,len(losses1)), rec_losses1[epskip:len(losses1)], label="no feedback", color=cmap(0.99), lw=3)
ax.plot(range(epskip,len(losses1)), rec_losses2[epskip:len(losses1)], label="with feedback", color=cmap(0.15), lw=3)

ax.set_xlabel("number of epochs")
ax.set_ylabel("loss")
plt.title("reconstruction loss")
ax.set_yscale("log")
ax.set_xticks([epskip,500,1000])
ax.set_yticks([50,100,200],["50","100","200"])
plt.ylim(40,250)
if save_autoenc_lossplots:
    plt.savefig(file_loc+f"autoencoder/nonlinear_reconstruction_loss_zeta{zeta_lossplot}_factor1.svg",format="svg")
    plt.savefig(file_loc+f"autoencoder/nonlinear_reconstruction_loss_zeta{zeta_lossplot}_factor1.png",format="png")



#Plot sparsity losses with and without student feedback
fig,ax = plt.subplots()
ax.xaxis.set_tick_params(width=5, length=10)
ax.yaxis.set_tick_params(width=5, length=10)

ax.plot(range(epskip,len(losses1)), spar_losses1[epskip:len(losses1)], label="no feedback", color=cmap(0.99), lw=3)
ax.plot(range(epskip,len(losses1)), spar_losses2[epskip:len(losses1)], label="with feedback", color=cmap(0.15), lw=3)

ax.set_xlabel("number of epochs")
plt.title("sparsity loss")
ax.set_yscale("log")
ax.set_xticks([epskip,500,1000])
ax.set_yticks([20,50,100,200],["20","50","100","200"])
plt.ylim(20,200)
if save_autoenc_lossplots:
    plt.savefig(file_loc+f"autoencoder/nonlinear_sparsity_loss_zeta{zeta_lossplot}_factor1.svg",format="svg")
    plt.savefig(file_loc+f"autoencoder/nonlinear_sparsity_loss_zeta{zeta_lossplot}_factor1.png",format="png")



#Plot SAE losses with and without student feedback plus goal finding losses
fig,ax = plt.subplots()
ax.xaxis.set_tick_params(width=5, length=10)
ax.yaxis.set_tick_params(width=5, length=10)

ax.plot(range(epskip,len(losses1)), (1-gamma_sparse)*rec_losses1[epskip:len(losses1)]+gamma_sparse*spar_losses1[epskip:len(losses1)], label="no feedback", color=cmap(0.99), lw=3)
ax.plot(range(epskip,len(losses1)), (1-gamma_sparse)*rec_losses2[epskip:len(losses1)]+gamma_sparse*spar_losses2[epskip:len(losses1)], label="with feedback", color=cmap(0.15), lw=3)
ax.plot(range(epskip,len(losses1)), goal_losses2[epskip:len(losses1)], label="goal finding loss", color="green", lw=3)

ax.set_xlabel("number of epochs")
plt.title("SAE/goal finding losses")
ax.set_yscale("log")
ax.set_xticks([epskip,500,1000])
ax.set_yticks([1,10,100],["1","10","100"])
plt.ylim(0.5,250)
if save_autoenc_lossplots:
    plt.savefig(file_loc+f"autoencoder/nonlinear_combined_loss_zeta{zeta_lossplot}_factor1.svg",format="svg")
    plt.savefig(file_loc+f"autoencoder/nonlinear_combined_loss_zeta{zeta_lossplot}_factor1.png",format="png")



#Plot SAE loss difference between cases with and without student feedback
plt.figure()
plt.ticklabel_format(style='plain')
axt = plt.gca()
axt.tick_params(width=5, length=10)

plt.plot(range(epskip,len(losses1)), (1-gamma_sparse)*rec_losses1[epskip:len(losses1)]+gamma_sparse*spar_losses1[epskip:len(losses1)]-((1-gamma_sparse)*rec_losses2[epskip:len(losses1)]+gamma_sparse*spar_losses2[epskip:len(losses1)]), label="loss difference", color="black", lw=3)
plt.plot(range(epskip,len(losses1)), np.zeros(len(losses1))[epskip:] , color="gray", linestyle="dashed",lw=3) #mark the 0 with a dashed line

plt.xlabel("number of epochs")
plt.xticks([epskip,500,1000])
plt.title("SAE loss difference")
plt.ylim(-65,5)
if save_autoenc_lossplots:
    plt.savefig(file_loc+f"autoencoder/nonlinear_lossdifference_zeta{zeta_lossplot}_factor1.svg",format="svg")
    plt.savefig(file_loc+f"autoencoder/nonlinear_lossdifference_zeta{zeta_lossplot}_factor1.png",format="png")
