import os
from sklearn.metrics import confusion_matrix
import matplotlib.pylab as plot

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
    with open("output_combine.txt", 'a') as f:  # risolvere bug qui
        print(str(len(inp[0])) + " " + str(len(inp[1])) + " " + str(len(inp[2])))
        for i in range(min(len(inp[0]), len(inp[1]), len(inp[2]))):
            gnb_i = inp[0][i]
            dt_i = inp[1][i]
            rf_i = inp[2][i]

            if rf_i != dt_i and rf_i == gnb_i:
                f.write(rf_i)
            else:
                f.write(dt_i)  # il decision tree è il classificatore con migliori performance
    f.close()
    print("majority voting: completato")


# procedura che restituisce la confusion matrix
def get_confusion_matrix():
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
    matrix = confusion_matrix(ground_truth, predictions, list(label_classes.keys()))
    print(matrix)
    fig = plot.figure()
    ax = fig.add_subplot(111)
    cax = ax.matshow(matrix)
    plot.title('Confusion matrix of the classifier')
    fig.colorbar(cax)
    ax.set_xticklabels([''] + list(label_classes.values()))
    ax.set_yticklabels([''] + list(label_classes.values()))
    plot.xlabel('Predicted')
    plot.ylabel('True')
    plot.show()


def main():
    # prepare_ground_truth()
    # merge_fold_predictions()
    # majority_voting_combiner()
    get_confusion_matrix()


if __name__ == '__main__':
    main()
