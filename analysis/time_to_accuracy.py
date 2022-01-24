import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


def create_plot(path, dataset, counter):
    time = list()
    mrr = list()
    ms = ["MTransE", "RotatE", "RDGCN", "MultiKE", "RREA(basic)", "KDCoE", "RREA(semi)"]
    for method in ms:
        time_temp = list()
        mrr_temp = list()
        parsed_path = path + dataset + "/" + method + "_" + dataset + "_cont_parsed"
        with open(parsed_path, "r") as fp:
            for line in fp:
                time_temp.append(float(line.split("\t")[0]))
                mrr_temp.append(float(line.split("\t")[1].rstrip()))
        time.append(time_temp)
        mrr.append(mrr_temp)
    thresholds = []
    for n in range(0, len(ms)):
        threshold = float(mrr[n][-1]) * 0.90
        nearest = min(mrr[n], key=lambda x: abs(float(x) - threshold))
        epoch = mrr[n].index(nearest)
        ti = time[n][epoch]
        thresholds.append((float(nearest), float(ti), epoch + 1, threshold))
    fig, ax = plt.subplots(figsize=(13, 8))
    plt.xlabel('Training Time (in log scale)')
    plt.ylabel('MRR')

    for z in range(7):
        time[z].insert(0, time[z][0])
        mrr[z].insert(0, float(0))

    ax.plot(time[0], mrr[0], color="green", linestyle="-", label="MTransE", linewidth=1.5)
    ax.plot(time[1], mrr[1], color="blue", linestyle="-", label="RotatE", linewidth=1.5)
    ax.plot(time[2], mrr[2], color="orange", linestyle="-", label="RDGCN", linewidth=1.5)
    ax.plot(time[3], mrr[3], color="pink", linestyle="-", label="MultiKE", linewidth=1.5)
    ax.plot(time[4], mrr[4], color="black", linestyle="-", label="RREA basic", linewidth=1.5)
    ax.plot(time[5], mrr[5], color="grey", linestyle="-", label="KDCoE", linewidth=1.5)
    ax.plot(time[6], mrr[6], color="purple", linestyle="-", label="RREA semi", linewidth=1.5)
    ax.legend(loc='upper left', prop={'size': 18})

    points = ""
    c = 0
    colors = ["green", "blue", "orange", "pink", "black", "grey", "purple"]
    for thres in thresholds:

        plt.plot(thres[1], thres[0], "o", color=colors[c], markersize=8)
        if c == 4:
            ax.annotate("(" + str(thres[2]) + "," + "{:.2f}".format(thres[0]) + ")", (thres[1], thres[0] - 0.05),
                        fontsize=16)
        else:
            ax.annotate("(" + str(thres[2]) + "," + "{:.2f}".format(thres[0]) + ")", (thres[1], thres[0] - 0.03),
                        fontsize=16)
        c += 1

    ax.set_ylim(ymin=0)
    ax.set_ylim(ymax=1)
    plt.margins(0)
    ax.set_xscale('log')

    plt.setp(ax.get_xticklabels(), rotation=0, horizontalalignment='right')
    plt.rcParams.update({'font.size': 18})
    ax.tick_params(axis='x', which='major', pad=8)
    plt.savefig("test_time_to_accuracy" + str(counter) + ".png")


def coefficient_variation(path):
    cv = lambda x: np.std(x, ddof=1) / np.mean(x)
    ms = ["MTransE", "RotatE", "RDGCN", "MultiKE", "RREA(basic)", "KDCoE", "RREA(semi)"]
    datasets = ["D_W_15K_V1", "D_W_15K_V2", "D_Y_15K_V1", "D_Y_15K_V2"]
    df = pd.DataFrame(columns=ms)
    ind = 0
    for dataset in datasets:
        cv_list = list()
        for method in ms:
            time_CV = list()
            flag = False
            parsed_path = path + dataset + "/" + method + "_" + dataset + "_cont_parsed"
            with open(parsed_path, "r") as fp:
                for line in fp:
                    if not flag:
                        time = float(line.split("\t")[0])
                        flag = True
                    else:
                        time = float(line.split("\t")[0]) - temp_time
                    temp_time = float(line.split("\t")[0])
                    time_CV.append(time)
            cv_list.append(cv(time_CV))
        df.loc[ind] = cv_list
        ind += 1
    return df


path = "TTA/"
ds = ["D_W_15K_V1", "D_W_15K_V2", "D_Y_15K_V1", "D_Y_15K_V2"]

counter = 0
for dataset in ds:
    create_plot(path, dataset, counter)
    counter += 1
print("Coefficient Variation")
print(coefficient_variation(path))