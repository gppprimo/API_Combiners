import os
from sklearn.metrics import confusion_matrix, accuracy_score, classification_report
import matplotlib.pylab as plot
import random
import numpy as np

folder_names = {"output_ML_GNB": "gaussian_naive_bayes", "output_ML_DT": "decision_tree",
                "output_ML_RF": "taylor2016appscanner_RF"}
folds_number = 10
path = "./Data/"
path_to_file = "/predictions/fold_"
combiner_input_file_names = ["combined_input_gaussian_naive_bayes.txt", "combined_input_decision_tree.txt",
                             "combined_input_taylor2016appscanner_RF.txt"]

label_classes = {
    0: 'BENIGN',
    1: 'Bot',
    2: 'DDoS',
    3: 'DoS GoldenE',
    4: 'DoS Hulk',
    5: 'DoS Slowhtt',
    6: 'DoS slowlor',
    7: 'FTP-Patator',
    8: 'Heartbleed',
    9: 'Infiltratio',
    10: 'PortScan',
    11: 'SSH-Patator',
    12: 'Web Attack'
}


# funzione che costruisce il ground_truth utilizzato per il calcolo
# delle performance del classificatore
def prepare_ground_truth():
    for i in range(1, 11, 1):
        ground_truth = []
        folder = list(folder_names.keys())
        if os.path.exists(path + folder[0] + path_to_file + str(i)):
            file_name = folder_names.get(folder[0]) + "_fold_" + str(i) + "_predictions.dat"
            f = open(path + folder[0] + path_to_file + str(i) + "/" + file_name, "r")
            dat_content = f.readlines()[1:]
            for line in dat_content:  # skip della prima riga del file che e' una stringa
                ground_truth.append(line.split()[0])
            f.close()
            with open("ground_truth.txt", 'a') as f:  # creo il nuovo file
                for j in ground_truth:
                    f.write(str(j) + '\n')
            f.close()
        else:
            print("Path " + path + folder[0] + path_to_file + str(i) + " non existent")

# merging dei file di interesse dei diversi fold dei singoli classificatori
def merge_fold_predictions():
    for k in range(0, 3, 1):  # per ogni cartella di output dei classificatori (#3)
        predictions = []  # inizializza la colonna da leggere
        for i in range(1, 11, 1):  # 10 cartelle fold
            folders = list(folder_names.keys())
            print(folders[k] + " - current folder: " + str(i))
            print(path + folders[k] + path_to_file + str(i))
            if os.path.exists(path + folders[k] + path_to_file + str(i)):  # path all'i-ma cartella fold
                file_name = folder_names.get(folders[k]) + "_fold_" + str(i) + "_predictions.dat"
                print(path + folders[k] + path_to_file + str(i) + "/" + file_name)
                f = open(path + folders[k] + path_to_file + str(i) + "/" + file_name, "r")
                dat_content = f.readlines()[1:]
                for line in dat_content:  # skip della prima riga del file che e' una stringa
                    predictions.append(line.split('	')[1])
                f.close()
                with open('combined_input_' + folder_names.get(folders[k]) + ".txt", 'a') as f:  # creo il nuovo file
                    for j in predictions:
                        f.write(str(j))
                f.close()
                predictions = []
            else:
                print("Path " + path + folders[k] + path_to_file + str(i) + " non existent")

# funzione che crea l'input da dare al combiner
def prepare_combiner_input():
    RF_input = []
    DT_input = []
    GNB_input = []
    combiner_input = [GNB_input, DT_input, RF_input]

    print("inizio generazione input")
    print(len(combiner_input_file_names))
    for i in range(len(combiner_input_file_names)):
        with open(combiner_input_file_names[i], 'r') as f:
            lines = f.readlines()
            for line in lines:
                combiner_input[i].append(line)
            print(len(combiner_input[i]))

    print("generazione input completato")
    return combiner_input


def majority_voting_combiner():
    inp = prepare_combiner_input()
    print("majority voting: iniziato")
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
    print("majority voting: completato")


# procedura che calcola la confusion matrix, la normalizza e imposta i parametri per il plot
def get_confusion_matrix(ground_truth, predictions):
    cm = confusion_matrix(ground_truth, predictions, list(label_classes.keys()))
    cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis] # normalizziamo
    print(cm)
    fig, ax = plot.subplots() #impostazioni varie per il plot
    im = ax.imshow(cm, interpolation='nearest', cmap=plot.cm.Blues)
    ax.figure.colorbar(im, ax=ax)
    # We want to show all ticks...
    ax.set(xticks=np.arange(cm.shape[1]),
           yticks=np.arange(cm.shape[0]),
           # ... and label them with the respective list entries
           xticklabels=list(label_classes.values()), yticklabels=list(label_classes.values()),
           title="Normalized Confusion Matrix",
           ylabel='True',
           xlabel='Predicted')

    # Rotate the tick labels and set their alignment.
    plot.setp(ax.get_xticklabels(), rotation=45, ha="right",
              rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    fmt = '.2f'
    thresh = cm.max() / 2.
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, format(cm[i, j], fmt),
                    ha="center", va="center",
                    color="white" if cm[i, j] > thresh else "black")
    fig.tight_layout()
    return ax


# procedura che restituisce un plot della confusion matrix e le misure di performance
def performance_measures():
    ground_truth = []
    predictions = []
    f = open("ground_truth.txt")
    lines = f.readlines()
    for line in lines:
        ground_truth.append(int(line.strip().split()[0]))  # conversione a int necessaria
    f.close()
    lines = []
    f = open("output_combine.txt")
    lines = f.readlines()
    for line in lines:
        predictions.append(int(line.strip().split()[0]))

    get_confusion_matrix(ground_truth, predictions)
    plot.show()

    print("Accuracy score: ", accuracy_score(ground_truth, predictions))
    print(classification_report(ground_truth, predictions))


def main():
    prepare_ground_truth()
    merge_fold_predictions()
    majority_voting_combiner()
    performance_measures()


if __name__ == '__main__':
    main()
