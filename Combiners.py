import math
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from Classifiers import taylor2016appscanner_RF, decision_tree, gaussian_naive_bayes
from Utility import *


def majority_voting_combiner():
    print("majority voting: iniziato")
    input = prepare_combiner_input()
    with open("output_combine.txt", 'a') as f:
        for i in range(len(input[0])):
            gnb_i = input[0][i]
            dt_i = input[1][i]
            rf_i = input[2][i]

            if rf_i != dt_i and rf_i == gnb_i:
                f.write(rf_i)
            elif rf_i != gnb_i and rf_i == dt_i:
                f.write(dt_i)
            elif rf_i != gnb_i and gnb_i == dt_i:
                f.write(dt_i)
            else:
                f.write(random.choice([gnb_i, dt_i, rf_i]))  # tie-breaking rule
    f.close()
    print("majority voting: completato.")
    print("Output file: 'output_combine.txt")


def weighted_majority_voting():

    # Ui = Di + |Ii| * ln(L - 1) + sum{Wk}, k from 1 to len(Ii)
    # L = numero di classi
    # Ui confidenza della classe i-ima
    # Ii subset dei classificatori che hanno deciso per la classe i
    # Di = ln(P(Ci)), P(Ci) probabilita' che la classe i-ima appaia nel Validation Set
    # Wk = ln(Pk/ (1 - Pk)) peso (accuracy) calcolato dal k-imo classificatore nel Validation Set

    num_classes = len(label_classes)
    log_classes: float = math.log(num_classes - 1)  # calcolo ln(L-1) che è costante
    fold_count = 1

    samples, categorical_lable_list = dataset_deserialized()
    samples = samples[0]
    categorical_labels = categorical_lable_list[0]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=124)  # 5_fold cross_validation
    for train_idx, test_idx in kfold.split(samples, categorical_labels):
        samples_train = samples[train_idx]
        samples_test = samples[test_idx]
        # split del training set in train e validation sets
        x_train, x_val, y_train, y_val = train_test_split(samples_train, categorical_labels[train_idx], shuffle="True",
                                                          random_state=124, test_size=0.5)

        occurrences_prob = []

        # calcolo di Di = log(P(Ci))
        for z in range(0, num_classes, 1):  # per ogni classe
            occurrences_prob.append(np.count_nonzero(y_val == z))   # Ci vedi il numero di occorrenze della classe i
            occurrences_prob[z] = occurrences_prob[z] / len(y_val)  # P(Ci) rapporta il valore trovato con
                                                                    # la dimensione del validation set
            occurrences_prob[z] = math.log(occurrences_prob[z])     # calcolo log(P(Ci))

        predict_rf = []
        predict_nb = []
        predict_dt = []

        # calcolo predizioni e valori di accuratezza sul validation set x i tre classificatori
        predict_rf, accuracy_rf = taylor2016appscanner_RF(x_val, y_val, x_train, y_train)
        predict_nb, accuracy_nb = gaussian_naive_bayes(x_val, y_val, x_train, y_train)
        predict_dt, accuracy_dt = decision_tree(x_val, y_val, x_train, y_train)

        print(accuracy_rf)
        print(accuracy_nb)
        print(accuracy_dt)

        # calcolo dei pesi Wi = log(pk/(1-pk)) dove pk è l'accuracy del k-esimo classificatore
        weights = []
        w_rf = math.log(accuracy_rf / (1 - accuracy_rf))
        w_nb = math.log(accuracy_nb / (1 - accuracy_nb))
        w_dt = math.log(accuracy_dt / (1 - accuracy_dt))

        weights.append(w_rf)
        weights.append(w_nb)
        weights.append(w_dt)

        predictions = []
        confidence = []
        decision = []
        sub_classes = 0

        for i in range(0, len(predict_dt), 1):  # per ogni riga di predizioni
            predictions.append(predict_rf[i])   # preleviamone la i-esima per tutti i classificatori
            predictions.append(predict_nb[i])
            predictions.append(predict_dt[i])
            for j in range(0, num_classes, 1):  # per ogni classe
                confidence.append(occurrences_prob[j])  # sommiamo la probabilità di occorrenza della
                                                        # classe j alla confidence
                for p in range(0, 3, 1):        # per ogni classificatore
                    if predictions[p] == j:     # se il classificatore ha predetto la classe j
                        confidence[j] += weights[p]  # aggiungiamo il peso
                        sub_classes += 1    # incrementiamo la dimensione del sotto-set di classificatori che
                                            # hanno scelto per la classe j
                confidence[j] += sub_classes * log_classes  # aggiungiamo |Ii|*log(L-1)
            wmv = [q for q, x in enumerate(confidence) if
                   x == max(confidence)]  # troviamo gli indici con valore massimo
            if len(wmv) > 1:  # se ne abbiamo più di uno
                decision.append(random.choice(wmv))  # tie - breaking rule
            else:
                decision.append(wmv)
            predictions = []  # puliamo le liste e parametri
            confidence = []
            sub_classes = 0
        with open("fold_" + str(fold_count) + "_predictions_wmv.txt", 'a') as f:  # creo il nuovo file
            for k in decision:
                f.write(str(k[0]) + '\n')  # metto le decisioni prese dal wmv
        f.close()
        fold_count += 1


def main():
    # prepare_ground_truth()
    # merge_fold_predictions()
    # majority_voting_combiner()
    # performance_measures()
    weighted_majority_voting()


if __name__ == '__main__':
    main()
