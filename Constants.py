folder_names = {"output_ML_GNB": "gaussian_naive_bayes", "output_ML_DT": "decision_tree",
                "output_ML_RF": "taylor2016appscanner_RF"}
path = "../"    # ../ - .Data/
path_to_file = "/predictions/fold_"
combiner_input_file_names = ["combined_input_gaussian_naive_bayes.txt", "combined_input_decision_tree.txt",
                             "combined_input_taylor2016appscanner_RF.txt"]
path_to_pickle = "CICIDS2017_corrected_76.pickle"

folds_number = 10

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