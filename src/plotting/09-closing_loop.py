#@title 22. PLOTS - Student performance plots (different goal locations trained), (Figs, 4e-h, 5b-e, S3-5)


#label dictionaries for all trained and unknown mazes respectively
label_dict_train=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstraining_4x4.pkl")
label_dict_test=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstest_4x4.pkl")

#plot settings
plt.rc('axes', labelsize=36)
plt.rc('xtick', labelsize=36)
plt.rc('ytick', labelsize=36)
plt.rc('axes', titlesize=36)
cmap=plt.get_cmap("bwr")

#a function to put p-value stars over bars
def add_pvalues(rects_x, rects_y, rect_width, pvalues, critical_p=0.05):
    '''
    rects_x - list of x-positions of the bars
    rects_y - list of bar heights (plus error bar)
    rect_width - rectangle width
    pvalues - list of pvalues (in order p01,p02,...,p0n,p12,...p1n,...,p n-1 n)
    critical_p - the critical p-value (if we are below that, we put a star in the plot)
    '''
    line_y = []
    y_offset_init=3
    y_offset_between=8
    ycounter=0 #counter for how many star bars we have already placed
    pcounter=0
    for i, [rect1_x, rect1_y] in enumerate(zip(rects_x, rects_y)):
        for j, [rect2_x, rect2_y] in enumerate(zip(rects_x, rects_y)):
            if i < j:
                pvalue = pvalues[pcounter]
                pcounter+=1
                star = ''
                if pvalue < critical_p:
                    star='*'
                if star:
                    x = (rect1_x  + rect2_x) / 2
                    y = max(rect1_y,rect2_y) + y_offset_init+ycounter*y_offset_between
                    line_y.append(y)
                    ax.annotate(star, xy=(x, y), fontsize=20,ha='center')
                    ax.plot([rect1_x , rect2_x],[y, y], lw=3, c='black')
                    ycounter+=1



for method in ["no_learning"]: #iterate over learning methods
    method_add_on="simple learning" if method=="simple_learning" else "no learning"
    for solving_rate_method in ["stepfactor"]:
        param= stepfactor_goalloc_plots if solving_rate_method=="stepfactor" else rdrate_goalloc_plots
        #potentially chuck out languages if the performances are bad
        chucked_indices=[[] for z in goal_groups_plots]
        for plot in [0,1,2,3]: #iterate over performance on known and unknown tasks

            goal_groups_adjusted=list(goal_groups_plots)
            if plot in [1,3] and 0 in goal_groups_plots:
                goal_groups_adjusted.remove(0)
            goal_groups_adjusted=np.array(goal_groups_adjusted)

            ldict=label_dict_train if plot in [0,1] else label_dict_test
            known_addon="trained" if plot in [0,1] else "unknown"
            avg_rates=[[],[],[],[]] if solving_rate_method=="stepfactor" else [[],[]] #average solving rates over all tasks
            sem_rates=[[],[],[],[]] if solving_rate_method=="stepfactor" else [[],[]] #standard error of the mean in the average solving rates over all tasks
            #for t-tests
            p01_list,p02_list,p03_list,p12_list,p13_list,p23_list=[],[],[],[],[],[]
            plists=[p01_list,p02_list,p03_list,p12_list,p13_list,p23_list]
            for m in goal_groups_adjusted:

                folder=folders_goalloc_plots[m]+"_known" if plot in [0,1] else folders_goalloc_plots[m]+"_unknown"

                rates=[[],[],[],[]] if solving_rate_method=="stepfactor" else [[],[]] #solving rates from all tasks listed for the three students (random walker is separate)

                for language in range(language_nr_goalloc_plots):
                    single_rates=[[],[],[],[]] if solving_rate_method=="stepfactor" else [[],[]]
                    #load the solving rates
                    for k,student in enumerate(["info","misinfo"]):
                        single_rates[k]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_{student}_{method}_{solving_rate_method}{param}.txt")
                    if solving_rate_method=="stepfactor":
                        single_rates[2]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalkersmart_{solving_rate_method}{param}.txt")
                        single_rates[3]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalker_{solving_rate_method}{param}.txt")

                    #filter out only the solving rates from tasks with trained goal locations
                    if plot==0 or plot==2:
                        for std in range(len(single_rates)):
                            newrates=[]
                            i=0
                            for j,label in ldict.items():
                                if label[2]!=student_init:
                                    if label[2] in train_goals_dict[m]:
                                        newrates+=[single_rates[std][i]]
                                    i+=1
                            single_rates[std]=newrates

                    #filter out only the solving rates from tasks with unknown goal locations
                    if plot==1 or plot==3:
                        for std in range(len(single_rates)):
                            newrates=[]
                            i=0
                            for j,label in ldict.items():
                                if label[2]!=student_init:
                                    if not(label[2] in train_goals_dict[m]):
                                        newrates+=[single_rates[std][i]]
                                    i+=1
                            single_rates[std]=newrates

                    #chuck the language out if the informed student is worse than the misinformed student!
                    if chuck_out:
                        if (plot==0 and (np.mean(single_rates[0])<np.mean(single_rates[3]) or np.mean(single_rates[0])<np.mean(single_rates[1]))) or (language in chucked_indices[m]):
                            if plot==0:
                                chucked_indices[m]+=[language]
                                print("Chucked out a language!")
                            continue

                    for index in range(len(single_rates)):
                        rates[index]+=[np.mean(single_rates[index])]

                print(f"goal location group {m}, goal finding rates for the single languages (in %)")
                print(f"info:            {np.round(100*np.array(rates[0]),1)} - average {round(100*np.mean(rates[0]),1)}")
                print(f"misinfo:         {np.round(100*np.array(rates[1]),1)} - average {round(100*np.mean(rates[1]),1)}")
                print(f"smart rd walker: {round(100*rates[2][0],1)}")
                print(f"rd walker:       {round(100*rates[3][0],1)}")

                #do t-tests (informed and misinformed with all others - for random walker comparison do 1-sided t-test, for student comparison do 2-sided t-test)
                #informed with misinformed
                p01=sp.stats.ttest_ind(rates[0], rates[1]).pvalue
                #informed with rd walkers
                p02=sp.stats.ttest_1samp(rates[0], rates[2][0]).pvalue
                p03=sp.stats.ttest_1samp(rates[0], rates[3][0]).pvalue
                #misinformed with rd walkers
                p12=sp.stats.ttest_1samp(rates[1], rates[2][0]).pvalue
                p13=sp.stats.ttest_1samp(rates[1], rates[3][0]).pvalue
                pvalues=[p01,p02,p03,p12,p13,1]
                #append p-values to list
                for i,plist in enumerate(plists):
                    plist+=[pvalues[i]]
                pinfo_array=np.round_(np.array([p01,p02,p03]),3)
                pmisinfo_array=np.round_(np.array([p12,p13]),3)
                print(f"p-values (t-test) for informed (vs misinfo, rd-smart, rd) are:    {pinfo_array}")
                print(f"p-values (t-test) for misinformed (vs rd-smart, rd) are: {pmisinfo_array}")
                print("")


                #add average solving rate and NOW STANDARD ERROR OF THE MEAN
                for j,rates_student in enumerate(rates):
                    avg_rates[j]+=[100*np.mean(rates_student)]
                    #std_rates[j]+=[100*np.std(rates_student)]
                    sem_rates[j]+=[100*np.std(rates_student, ddof=1)/mt.sqrt(language_nr_goalloc_plots)]

            #plot
            fig,ax=plt.subplots(figsize=(28,4.8))
            plt.tick_params(bottom=False, labelbottom=False) #remove ticks and labels from y axis
            plt.grid(visible=True, axis="y")
            plt.rcParams['svg.fonttype']='none' #"to make later editing of figures easier" (Carlos)
            axt = plt.gca()
            axt.tick_params(width=5, length=10)
            labels=["informed student","misinformed student","smart random walker","random walker"] if solving_rate_method=="stepfactor" else ["informed student","misinformed student"]
            colors=[cmap(0.15),cmap(0.25),cmap(0.99),cmap(0.75)] if solving_rate_method=="stepfactor" else [cmap(0.15),cmap(0.25)]
            bar_positions= [goal_groups_adjusted-0.2, goal_groups_adjusted+0.2] if solving_rate_method=="rate" else [goal_groups_adjusted-0.18, goal_groups_adjusted-0.06,goal_groups_adjusted+0.06,goal_groups_adjusted+0.18]
            for rates,sem,label,color,bar_pos in zip(avg_rates, sem_rates, labels, colors, bar_positions):
                rates,sem=np.array(rates),np.array(sem)
                width =0.1 if solving_rate_method=="stepfactor" else 0.35
                ax.bar(bar_pos,rates,label=label, width=width, color=color)
                if label in ["informed student","misinformed student"]:
                    ax.errorbar(bar_pos,rates,yerr=sem, capsize=6,elinewidth=4,capthick=3,color="black", fmt="none")

            trange=range(7) if plot in [0,2] else range(6)
            rect_width =0.1 if solving_rate_method=="stepfactor" else 0.35
            for t in trange:
                rects_x=[t-0.18,t-0.06,t+0.06,t+0.18] if plot in [0,2] else [t+1-0.18,t+1-0.06,t+1+0.06,t+1+0.18]
                rects_y=[avg_rates[i][t]+sem_rates[i][t] for i in range(4)]
                #only show stars if informed is better, not other way around!
                p01val=p01_list[t] if avg_rates[0][t]>avg_rates[1][t] else 1
                p02val=p02_list[t] if avg_rates[0][t]>avg_rates[2][t] else 1
                p03val=p03_list[t] if avg_rates[0][t]>avg_rates[3][t] else 1
                p12val=p12_list[t] if avg_rates[1][t]>avg_rates[2][t] else 1
                p13val=p13_list[t] if avg_rates[1][t]>avg_rates[3][t] else 1
                p_values=[p01val, p02val,1,1,1,1]
                add_pvalues(rects_x,rects_y,rect_width,p_values)


            ax.set_ylim(0,120)
            ax.set_ylabel("tasks solved (%)")
            if plot==0:
                plt.title(f"student evaluated at trained goals for trained mazes (0 and 1 wall states)")
            elif plot==1:
                plt.title(f"student evaluated at unknown goals for trained mazes (0 and 1 wall states)")
            elif plot==2:
                plt.title(f"student evaluated at trained goals for unknown mazes (2 wall states)")
            elif plot==3:
                plt.title(f"student evaluated at unknown goals for unknown mazes (2 wall states)")
            ax.set_xlim(-0.25,len(goal_groups_plots)-0.25)
            ax.set_yticks([0,20,40,60,80,100])


            ax.set_xlabel(r"goal locations trained")
            #plt.legend(bbox_to_anchor=(1,1), loc="upper left", fontsize=25) #put legend outside the plot so that no lines are covered!
            if save_goalloc_plots:
                plt.savefig(file_loc+"student/"+f"{folders_goalloc_plots[0]}"+f"/solvingrates_{method}_{solving_rate_method}_zeta5_factor1_plot{plot}.png", bbox_inches='tight', format="png")
                plt.savefig(file_loc+"student/"+f"{folders_goalloc_plots[0]}"+f"/solvingrates_{method}_{solving_rate_method}_zeta5_factor1_plot{plot}.svg", format="svg")
            plt.show()

