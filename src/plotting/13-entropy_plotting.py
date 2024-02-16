#@title 27. PLOTS - Entropy binning approach (Fig. 3e)
'''
The plots of the entropy analysis
'''

cmap=plt.get_cmap("bwr")

#two plots for the two different histogram methods
for hplot, method in zip([1], ["equal extent"]):
    #I figured out that method 2 is the better one (always lower fit errors, more reasonable...)
    if hplot==0:
        continue
    #calculate mean and standard deviations over the 5 languages
    #change hear what should be plotted (also have to change colors and labels below of course)
    H_arrs=[H_feedback, H_nofeedback, H_pstd, H_feedback_noregu, H_pstd_noregu]
    H_means, H_stds=[], []
    for H_arr in H_arrs:
        H_stds+=[np.std(H_arr[hplot], axis=1)]
        H_means+=[np.mean(H_arr[hplot], axis=1)]

    #plot theoretical maximum entropy as dashed line
    Hmax=sp.stats.entropy([1/len(q_matrix_dict)]*len(q_matrix_dict))
    plt.plot([0,64], [Hmax, Hmax], ls="--", color="black", label="maximum possible entropy", lw=3)
    #scatter plots of entropies and corresponding hyperbolic tangent curve fits
    H_fitlist=[H_qteacher[hplot]]+H_means
    Hstd_fitlist=[None]+H_stds
    colors=["black", cmap(0.12), cmap(0.99), cmap(0.35), cmap(0.12), cmap(0.35)]
    labels=["teacher Q-matrices", "messages w/ feedback", "messages w/o feedback","student action prob. matrices", "messages w/ feedback (no rec.)", "student action prob. matrices (no rec.)"]
    markers=["o", "D", "s", "D", "x", "x"]
    err=0
    for H, H_std, color, label, marker in zip(H_fitlist, Hstd_fitlist, colors, labels, markers):
        #popt, pcov, infodict, _, _=curve_fit(lambda x, a, c: tanhfunc(x, a, c, Hmax), H_binlist, H, p0=[0.5,0.5], sigma=H_std, absolute_sigma=True, full_output=True)
        #err+=np.sum(np.absolute(infodict["fvec"]))
        #aparam, cparam=round(popt[0],3), round(popt[1],3)
        #print(aparam, cparam)
        #plt.plot(H_binlist, tanhfunc(H_binlist, popt[0], popt[1], Hmax), color=color, lw=3)
        plt.scatter(H_binlist, H, marker=marker, s=75, color=color, label=label, lw=3)
        #plt.errorbar(H_binlist, H, yerr=H_std, fmt=marker, ms=8, color=color, label=label, lw=3)

    #print(f"error is {err}")

    plt.legend(loc=4, fontsize=11)

    plt.xlim(0,64)
    plt.ylim(2,5.5)
    plt.xlabel(r"bins per PC-axis ($\it{n}$)")
    plt.ylabel("Shannon entropy")
    plt.title("entropy at different comm. stages", size=28, y=1.05)

    plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
    plt.rc('axes', labelsize=28)
    plt.rc('xtick', labelsize=28)
    plt.rc('ytick', labelsize=28)

    plt.tick_params(width=5, length=10)

    plt.show()
