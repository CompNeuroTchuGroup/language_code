#-----------------------------------------------------------------------------------------------------------------------
#@title 2. INIT - Changeable parameters
#-----------------------------------------------------------------------------------------------------------------------




#generate a new "language" with the autoencoder, or load it from a file
language_code: str="nonlinear_goallocs0_zeta5_language2" #code to get autoencoder parameters from a file

#closing the loop
language_code_closingloop="nonlinear_goallocs0_zeta5_language0" #language code for which language we should plot the results of "closing the loop"

#autoencoder loss plots
zeta_lossplot=5 #change the hyperparameter zeta here to see the loss plots for different values (1,2,5,10)

#plot the student performances for different groups of goal locations trained
folders_goalloc_plots=[f"nonlinear_goallocs{i}_zeta5_factor1" for i in range(7)] #the goal groups 0-6 have the same order as in the plots in the paper
language_nr_goalloc_plots=5 #how many languages per group of goal locations?
chuck_out=False #chuck out languages in which the performance of the informed student is below the misinformed student or below the random walker?

#read Q-matrices from a file
qmat_read_code: str="training_4x4" #either training4x4 or test4x4