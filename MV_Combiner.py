# Majority Voting

import os

folder_names = {"output_ML_DT": "decision_tree", "output_ML_GNB": "gaussian_naive_bayes",
                "output_ML_RF": "taylor2016appscanner_RF"}
folds_number = 10
path = "../"
path_to_file = "/predictions/fold_"
combiner_input_file_names = ["combined_input_decision_tree.txt", "combined_input_gaussian_naive_bayes.txt", "combined_input_taylor2016appscanner_RF.txt"]

# funzione che fa il merge di tutti i file "prediction" di ogni fold di ogni classificatore
def merge_fold_predictions():
    column_read = []
    for folder in folder_names.keys():  # per ogni cartella di output dei classificatori (#3)
        for i in range(1, 11, 1):  # 10 cartelle fold
            print(folder + " - current folder: " + str(i))
            if os.path.exists(path + folder + path_to_file + str(i)):  # path all'i-ma cartella fold
                list_file = os.listdir(path + folder + path_to_file + str(i))  # lista dei file nella fold_i
                token = "fold" + "_" + str(i) + "_predictions"  # stringa per selezionare il file corretto
                file_name = list(filter(lambda s: token in s, list_file))[0]
                datContent = [i.strip().split()
                              for i in open(path + folder + path_to_file + str(i)
                                            + "/" + file_name).readlines()]  # leggo il file e mi salvo la seconda colonna
                for line in datContent[1:]:  # skip della prima riga del file che e' una stringa
                    column_read.append(line[1])
                with open('combined_input_' + folder_names[folder] + ".txt", 'a') as f:  # creo il nuovo file
                    for i in column_read:
                        f.write(str(i) + '\n')

def prepare_combiner_input():
    RF_input = []
    DT_input = []
    GNB_input = []
    Combiner_Input = [RF_input, DT_input, GNB_input]

    print("inizio generazione input")

    for i in range(len(combiner_input_file_names)):
        with open(combiner_input_file_names[i], 'r') as f:
            Combiner_Input[i].append(f.readlines())

    print("generazione input completato")
    return Combiner_Input

# funzione che implementa il majority voting combiner
#   - input: due o piu' file corrispondenti agli output dei classificatori
#   - output: file contenentie le scelte del combinatore
def majority_voting_combiner(input):
    print("majority voting: iniziato")
    with open("output_combine.txt", 'a') as f: # risolvere bug qui
        print(str(len(input[0])) + " " + str(len(input[2])) + " " + str(len(input[3])))
        for i in range(min(len(input[0]), len(input[2]), len(input[3]))):
            rf_i = input[0][i]
            dt_i = input[1][i]
            gnb_i = input[2][i]

            if rf_i != dt_i and dt_i == gnb_i:
                f.write(dt_i)
            else:
                f.write(rf_i)
    f.close()
    print("majority voting: completato")


def main():
    #merge_fold_predictions()
    majority_voting_combiner(prepare_combiner_input())


if __name__ == '__main__':
    main()
