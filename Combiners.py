import math
import random
import numpy as np
from sklearn.model_selection import StratifiedKFold, train_test_split
from Classifiers import taylor2016appscanner_RF, decision_tree, gaussian_naive_bayes
from Utility import *


def majority_voting_combiner():
    print("majority voting: iniziato")
    inp = prepare_combiner_input()
    with open("output_combine.txt", 'a') as f:
        for i in range(len(inp[0])):
            gnb_i = inp[0][i]
            dt_i = inp[1][i]
            rf_i = inp[2][i]

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

    # Ui confidenza della classe i-ima
    # L = numero di classi
    # Ii subset dei classificatori che hanno deciso per la classe i
    # Di = ln(P(Ci)), P(Ci) probabilita' che la classe i-ima appaia nel Validation Set
    # Wk = ln(Pk/ (1 - Pk)) peso (accuracy) calcolato dal k-imo classificatore nel Validation Set

    num_classes = len(label_classes)    # L
    log_classes: float = math.log(num_classes - 1)  # ln(L-1)
    fold_count = 1

    samples, categorical_lable_list = dataset_deserialization()
    samples = samples[0]
    categorical_labels = categorical_lable_list[0]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=124)  # 5_fold cross_validation
    # con 5 split, la dimensione del test_set è il 20% del totale
    for train_idx, test_idx in kfold.split(samples, categorical_labels):
        samples_train = samples[train_idx]
        samples_test = samples[test_idx]
        # split del training set in train e validation sets
        # test_size = 0.25 in modo che il validation set sia delle stesse dimensioni del test set
        x_train, x_val, y_train, y_val = train_test_split(samples_train, categorical_labels[train_idx], shuffle="True",
                                                          random_state=124, test_size=0.25)

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

        # calcolo predizioni e valori di accuratezza sul validation set per i tre classificatori
        predict_rf, accuracy_rf = taylor2016appscanner_RF(x_val, y_val, x_train, y_train,
                                                          samples_test, categorical_labels[test_idx])
        predict_nb, accuracy_nb = gaussian_naive_bayes(x_val, y_val, x_train, y_train,
                                                       samples_test, categorical_labels[test_idx])
        predict_dt, accuracy_dt = decision_tree(x_val, y_val, x_train, y_train,
                                                samples_test, categorical_labels[test_idx])

        print(accuracy_rf)
        print(accuracy_nb)
        print(accuracy_dt)

        # calcolo dei pesi Wk = log(pk/(1-pk)) dove pk è l'accuracy del k-esimo classificatore
        weights = []
        w_rf = math.log(accuracy_rf / (1 - accuracy_rf))
        w_nb = math.log(accuracy_nb / (1 - accuracy_nb))
        w_dt = math.log(accuracy_dt / (1 - accuracy_dt))

        weights.append(w_rf)
        weights.append(w_nb)
        weights.append(w_dt)

        predictions = []
        confidence = []
        decisions = []
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
                decisions.append(random.choice(wmv))  # tie - breaking rule
            else:
                decisions.append(wmv)
            predictions = []  # puliamo le liste e parametri
            confidence = []
            sub_classes = 0
        write_predictions(categorical_labels[test_idx], decisions, "wmv", fold_count) # scrittura predizioni
        print("Accuracy score fold_" + str(fold_count) + ": ", accuracy_score(categorical_labels[test_idx], decisions))
        fold_count += 1


def recall_combiner():

    # Ui = Di + |Ii| * ln(L - 1) + sum{W_k,i}, k from 1 to len(Ii)

    # Ui confidenza della classe i-ima
    # L = numero di classi
    # Ii subset dei classificatori che hanno deciso per la classe i
    # Di = ln[P(Ci)] + sum(ln(1 - P_k,i)), k from 1 to num_classifiers], P(Ci) probabilita' che la classe i-ima appaia
    # nel Validation Set
    # Wk = ln(P_k,i/ (1 - P_k,i)) peso (recall) calcolato dal k-imo classificatore sull'i-esima classe nel
    # Validation Set

    num_classes = len(label_classes)
    log_classes: float = math.log(num_classes - 1)  # calcolo ln(L-1) che è costante
    fold_count = 1

    samples, categorical_lable_list = dataset_deserialization()
    samples = samples[0]
    categorical_labels = categorical_lable_list[0]
    kfold = StratifiedKFold(n_splits=5, shuffle=True, random_state=124)  # 5_fold cross_validation
    # con 5 split, la dimensione del test_set è il 20% del totale
    for train_idx, test_idx in kfold.split(samples, categorical_labels):
        samples_train = samples[train_idx]
        samples_test = samples[test_idx]
        # split del training set in train e validation sets
        # test_size = 0.25 in modo che il validation set sia delle stesse dimensioni del test set
        x_train, x_val, y_train, y_val = train_test_split(samples_train, categorical_labels[train_idx], shuffle="True",
                                                          random_state=124, test_size=0.25)

        occurrences_prob = []  # conterrà i ln[P(Ci)]
        sumoflogs = []  # conterrà la sum(ln(1 - P_k,i))
        delta = []

        predict_rf = []
        predict_nb = []
        predict_dt = []

        # calcolo predizioni e valori di accuratezza sul validation set x i tre classificatori
        predict_rf, accuracy_rf, recalls_rf = taylor2016appscanner_RF(x_val, y_val, x_train, y_train,
                                                                      samples_test, categorical_labels[test_idx])
        predict_nb, accuracy_nb, recalls_nb = gaussian_naive_bayes(x_val, y_val, x_train, y_train,
                                                                   samples_test, categorical_labels[test_idx])
        predict_dt, accuracy_dt, recalls_dt = decision_tree(x_val, y_val, x_train, y_train,
                                                            samples_test, categorical_labels[test_idx])

        print(accuracy_rf)
        print(accuracy_nb)
        print(accuracy_dt)

        # print(recalls_rf)

        weights = []
        w_rf = []
        w_nb = []
        w_dt = []

        # poiché c'è da calcolare il log(1 - P_k_i), è necessario che il valore di recall sia 0 < rec < 1
        # quindi dov'è 0 sostituiamo con il valore minimo, dov'è 1, con il valore massimo
        recalls_rf[recalls_rf == 0] = 0.00000001
        recalls_nb[recalls_nb == 0] = 0.00000001
        recalls_dt[recalls_dt == 0] = 0.00000001

        recalls_rf[recalls_rf == 1] = 0.99999999
        recalls_nb[recalls_nb == 1] = 0.99999999
        recalls_dt[recalls_dt == 1] = 0.99999999

        # calcolo di Di = log(P(Ci))
        for z in range(0, num_classes, 1):  # per ogni classe
            occurrences_prob.append(np.count_nonzero(y_val == z))  # Ci vedi il numero di occorrenze della classe i
            occurrences_prob[z] = occurrences_prob[z] / len(y_val)  # P(Ci) rapporta il valore trovato con
                                                                    # la dimensione del validation set
            occurrences_prob[z] = math.log(occurrences_prob[z])  # calcolo log(P(Ci))
            # calcolo la somma dei log(1 - P_k,i)
            sumoflogs.append(math.log(1 - recalls_rf[z]) + math.log(1 - recalls_dt[z]) + math.log(1 - recalls_nb[z]))
            # print(sumoflogs)
            delta.append(occurrences_prob[z] + sumoflogs[z])
            # print(delta)

        # print(recalls_rf)

        # poniamo nei pesi il valore di log[P_k,i/(1 - P_k,i)]

        for x in np.nditer(recalls_rf, op_flags=['readwrite']):
            w_rf.append(math.log(x/(1 - x)))
        for x in np.nditer(recalls_nb, op_flags=['readwrite']):
            w_nb.append(math.log(x/(1 - x)))
        for x in np.nditer(recalls_dt, op_flags=['readwrite']):
            w_dt.append(math.log(x/(1 - x)))

        weights.append(w_rf)
        weights.append(w_nb)
        weights.append(w_dt)

        predictions = []
        confidence = []
        decisions = []
        sub_classes = 0

        for i in range(0, len(predict_dt), 1):  # per ogni riga di predizioni
            predictions.append(predict_rf[i])  # preleviamone la i-esima per tutti i classificatori
            predictions.append(predict_nb[i])
            predictions.append(predict_dt[i])
            for j in range(0, num_classes, 1):  # per ogni classe
                confidence.append(delta[j])  # sommiamo la probabilità di occorrenza della
                # classe j alla confidence
                confidence[j] += math.log(1 - recalls_rf[j])
                confidence[j] += math.log(1 - recalls_nb[j])
                confidence[j] += math.log(1 - recalls_dt[j])
                for p in range(0, 3, 1):  # per ogni classificatore
                    if predictions[p] == j:  # se il classificatore ha predetto la classe j
                        confidence[j] += weights[p][j]  # aggiungiamo il peso
                        sub_classes += 1  # incrementiamo la dimensione del sotto-set di classificatori che
                        # hanno scelto per la classe j
                confidence[j] += sub_classes * log_classes  # aggiungiamo |Ii|*log(L-1)
            recall = [q for q, x in enumerate(confidence) if
                   x == max(confidence)]  # troviamo gli indici con valore massimo
            if len(recall) > 1:  # se ne abbiamo più di uno
                decisions.append(random.choice(recall))  # tie - breaking rule
            else:
                decisions.append(recall)
            predictions = []  # puliamo le liste e parametri
            confidence = []
            sub_classes = 0
        write_predictions(categorical_labels[test_idx], decisions, "rec", fold_count)  # scrittura predizioni
        print("Accuracy score fold_" + str(fold_count) + ": ", accuracy_score(categorical_labels[test_idx], decisions))
        fold_count += 1


def main():
    # prepare_ground_truth()
    # merge_fold_predictions()
    # majority_voting_combiner()
    # weighted_majority_voting()
    recall_combiner()


if __name__ == '__main__':
    main()
