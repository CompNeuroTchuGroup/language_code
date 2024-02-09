#@title 28. EXECUTION - Hyperparameter study (message length, student loss, number of epochs): extract rates

'''
Hyperparameter analysis - we kept everything constant and varied only one of the three parameters mentioned in the title
'''

#label dictionaries for all trained and unknown mazes respectively
label_dict_train=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstraining_4x4.pkl")
label_dict_test=read_dict_from_pkl(file_loc+"teacher/label dictionaries/"+f"q_matrices_labelstest_4x4.pkl")

for folder_tag, xaxis in zip(["epochs", "zeta", "mlength_K"], [[100,200,400,600,800,1000], [1,2,5,10,20], [1,2,3,4,5,6,7,8]]):

    folders_known=[f"nonlinear_{folder_tag}{i}_known" for i in xaxis]
    folders_unknown=[f"nonlinear_{folder_tag}{i}_unknown" for i in xaxis]

    language_nr_hyp=5
    param=2 #the stepfactor
    #potentially chuck out languages if the performances are bad
    for plot, ldict, known_addon, folders in zip([0,1], [label_dict_train, label_dict_test], ["known","unknown"], [folders_known, folders_unknown]): #iterate over performance on known and unknown tasks

        for j,folder in enumerate(folders):
            rates=[[],[],[],[]]

            for language in range(language_nr_hyp):
                single_rates=[[],[],[],[]]
                #load the solving rates
                for k,student in enumerate(["info","misinfo"]):
                    single_rates[k]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_{student}_no_learning_stepfactor2.txt")
                single_rates[2]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalkersmart_stepfactor2.txt")
                single_rates[3]=np.loadtxt(file_loc+"student/"+f"{folder}/{folder}_language{language}/"+f"solving_rates_rdwalker_stepfactor2.txt")

                for index in range(len(single_rates)):
                    rates[index]+=[np.mean(single_rates[index])]

            np.savetxt(file_loc+"hyperparameters/"+f"rates {folder_tag} {j} {known_addon}", np.array(rates))
