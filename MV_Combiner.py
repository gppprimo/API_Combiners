# Majority Voting

import os
import pickle

folder_names = {"output_ML_DT": "decision_tree", "output_ML_GNB": "gaussian_naive_bayes",
                "output_ML_RF": "taylor2016appscanner_RF"}
folds_number = 10
path = "../"
path_to_file = "/predictions/fold_"


def majority_voting_combiner():
    column_read = []
    for folder in folder_names.keys():  # per ogni cartella di output dei classificatori (#3)
        for i in range(1, 11, 1):   # 10 cartelle fold
            print(folder + " - current folder: " + str(i))
            if os.path.exists(path + folder + path_to_file + str(i)):   #path all'i-ma cartella fold
                list_file = os.listdir(path + folder + path_to_file + str(i))   # lista dei file nella fold_i
                token = "fold" + "_" + str(i) + "_predictions"  #stringa per selezionare il file corretto
                file_name = list(filter(lambda s: token in s, list_file))[0]
                datContent = [i.strip().split()
                              for i in open(path + folder + path_to_file + str(i)
                                + "/" + file_name).readlines()] # leggo il file e mi salvo la seconda colonna
                for line in datContent[1:]: #skip della prima riga del file che e' una stringa
                    column_read.append(line[1])
                with open('combined_input_' + folder_names[folder] + ".txt", 'a') as f: # creo il nuovo file
                    for i in column_read:
                        f.write(str(i) + '\n')

        # calcolo major voting per ogni elemento del file combined_input

def main():
    majority_voting_combiner()


if __name__ == '__main__':
    main()
