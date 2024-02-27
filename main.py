from fixed_parameters import *
from changeable_parameters import *

from src.execution.q_matrix_generation import *
from src.execution.language_training import *
from src.execution.calculate_student_performance import *
from src.execution.task_vectors import *
from src.execution.entropy_binning import *

# Uncomment if GPU is to be used - right now use CPU, as we have very small networks and for them, CPU is actually faster
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Press the green button in the gutter to run the script.
if __name__ == '__main__':

    # 01 - q_matrix_generation
    print("Q-matrix generation...")
    qmat_save_code: str = "dummy"  # file for saving newly generated Q-matrices (if qmat_gen=True)
    q_matrix_dict, wall_state_dict, label_dict = generate_qmat(wall_state_dict, qmat_gen=True)
    label_dict = insert_shortest_path_length(label_dict)
    print("done!")

    # 02-language-training
    print("Language training...")
    message_dict = train_language(q_matrix_dict, label_dict, language_gen)
    print("done!")

    # ------------------------------------------------------------------------------------------------------------------
    # 03 - calculate_student_performance
    # ------------------------------------------------------------------------------------------------------------------

    # todo: do this also for 1 and 2
    random_repeat = 5  # randomness repetition for misinformed students
    rdwalker_base_rates = []  # baseline random walker rates we set for each task (values like 0.1 for 10% and 0.2 for 20%)
    stepfactors = [2]  # number of steps we allow for the student in terms of shortest path length for each task
    methods = [
        "no_learning"]  # could choose "no_learning" (action choice by percentages) and/or "simple_learning" (greedy action choices, but if you come back to a state you take the next best action)

    # "known" for trained worlds or "unknown" for untrained worlds
    if qmat_read_code == "training4x4":
        known_addon = "known"
    elif qmat_read_code == "test4x4":
        known_addon = "unknown"
    else:
        known_addon = None

    # Parameters specified by the languages we want to evaluate
    goal_locs = [0]  # list of trained goal locations

    language_nr_evaluation = 1  # number of languages to evaluate (assume it's the same for all different goal locations)
    language_codes = [f"test_language"]
    saving_folders = [f"test_language_{known_addon}"]
    evalute_nonlinear_ae, evaluate_nonlinear_std = True, True  # are the activations in the autoencoder/student nonlinear?

    if student_evaluate:
        print("Compute student performance...")
        evaluate_students(label_dict, q_matrix_dict)  # todo: this doesn't work on the notebook as well
        print("done!")
    else:
        print("Skipping student evaluation.")

    # ------------------------------------------------------------------------------------------------------------------
    # 10 - create task vectors and do topographical similarity analysis
    # ------------------------------------------------------------------------------------------------------------------
    print("Doing topographical analysis...")
    indicatorvector_dict, indicatorvector_dictw = create_goal_wall_indicator(label_dict,
                                                                             goal_reward=goal_reward,
                                                                             wall_reward=wall_reward)

    calculate_distance_between_message_pairs(q_matrix_dict, indicatorvector_dict, indicatorvector_dictw)
    print("done!")

    # ------------------------------------------------------------------------------------------------------------------
    # 12 - Calculate entropy of message and other distributions
    # ------------------------------------------------------------------------------------------------------------------
    print("Calculating entropies...")
    H_language_codes, Hm_arrs, Hq_arrs, Hp_arrs = compute_all_entropies(label_dict, q_matrix_dict)
    print("done!")

    # ------------------------------------------------------------------------------------------------------------------
    # 14 - Calculate entropy of message and other distributions
    # ------------------------------------------------------------------------------------------------------------------