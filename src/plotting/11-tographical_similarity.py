#@title 25. PLOTS - Topographical similarity analysis (Figs. 3b, c)
'''
#See previous block - here are the plots
#basically what we see is how far are the messages from each other, when the ground truth meanings
#(i.e. Q-matrices or proba matrices) have a certain distance from each other

#as every language is different, we also normalize by average message pair distance in the second plot
#and in both cases we compare 5 languages each

#left column: not normalized, right column: normalized by average message pair distance
#first row: Q-matrices
#second row: P-matrices
'''

fig, ax =plt.subplots(1,2, figsize=(14,5.6))
cmap=plt.get_cmap("bwr")
plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
plt.rc('axes', labelsize=28)
plt.rc('xtick', labelsize=28)
plt.rc('ytick', labelsize=28)

for i, (language_prefix, color, color2, label) in enumerate(zip(["nonlinear_nostudent", "nonlinear_goallocs0_zeta5"], [cmap(0.99), cmap(0.15)], [cmap(0.75), cmap(0.25)], ["no feedb.", "with feedb."])):
    for j, meaning in enumerate(["q","vw"]):

        yvals=np.zeros((5,n_bins_topo))

        for l,lcode_topo in enumerate([f"{language_prefix}_language{k}" for k in range(5)]):
            x, y=np.loadtxt(file_loc+ f"topographic similarity/{meaning} vs m {n_bins_topo} bins {norm_topo}-norm {lcode_topo}")
            if l==0:
                xvals=x
            yvals[l]=y

        #average across languages
        yvals, yvals_std =np.mean(yvals, axis=0), np.std(yvals, axis=0)

        #linear curve fits for both scenarios
        popt1, pcov1=curve_fit(linfunc, xvals, yvals, sigma=yvals_std, absolute_sigma=True)

        #parameter formatting
        m1, b1=np.round(popt1[0],3), np.round(popt1[1],3)
        m1_std, b1_std=np.round(np.sqrt(np.diagonal(pcov1))[0],3), np.round(np.sqrt(np.diagonal(pcov1))[1],3)

        #first plot: regular message distances (averaged over languages)
        ax[j].errorbar(xvals, yvals, yerr=yvals_std, c=color, fmt="o", ms=5, label=label+f": m={m1}"+u"\u00B1"+f"{m1_std}", lw=3)
        ax[j].plot(xvals, linfunc(xvals, *popt1), color=color, lw=3)
        ax[j].legend(loc=2, fontsize=18)


xlims=[[0,6.5], [0,9]] if norm_topo==2 else [[0,35],[0,15]] if norm_topo==1 else None
ylims =[0,1.2]
ax[0].set_xlabel(f"teacher Q-matrix distance")
ax[1].set_xlabel(f"spatial task distance")
ax[0].set_ylabel(f"message distance")
ax[1].set_ylabel(f"message distance")

for i in range(2):
    ax[i].set_xlim(xlims[i])
    ax[i].set_ylim(ylims)
    ax[i].tick_params(width=5, length=10)

ax[0].set_title("teacher meaning", size=28, y=1.05)
ax[1].set_title("spatial meaning", size=28, y=1.05)

fig.tight_layout() #only titles for columns and rows

plt.subplots_adjust(wspace=0.5, hspace=0.6)

plt.show()