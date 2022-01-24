import pandas as pd
import Orange
import matplotlib.pyplot as plt
from scipy import stats

# choose [FRIEDMAN or NEMENYI]
play = "NEMENYI"
# choose metric [Hits@1, Hits@10, MR, MRR]
metric = "Hits@1"

if play == "FRIEDMAN":
    data1 = pd.read_excel("performances_and_statistics.xlsx", usecols=range(0, 32), skiprows=0)
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)

    for c in data1.columns:
        data1[c] = data1[c].apply(pd.to_numeric, errors='coerce')
    data1 = data1.head(8)

    if metric == "Hits@1":
        cols = [col for col in data1.columns if 'hits@1' in col and 'hits@10' not in col]
    elif metric == "Hits@10":
        cols = [col for col in data1.columns if 'hits@10' in col]
    elif metric == "MR":
        cols = [col for col in data1.columns if 'MR' in col and 'MRR' not in col]
    elif metric == "MRR":
        cols = [col for col in data1.columns if 'MRR' in col]

    data1 = data1[cols]
    for c in cols:
        data1[c] = data1[c].fillna(0)

    data1 = data1.T
    arr = data1.to_numpy()

    _, pvalue = stats.friedmanchisquare(arr[0], arr[1], arr[2], arr[3], arr[4], arr[5], arr[6], arr[7])
    print(metric + " pvalue = " + str(pvalue))
elif play == "NEMENYI":

    col = list()
    data = pd.read_excel("performances_and_statistics.xlsx", usecols=range(0, 32), skiprows=0)


    for a in data.columns:
        col.append(a)

    for c in col:
        data[c] = data[c].apply(pd.to_numeric, errors='coerce')

    data = data.head(8)

    if metric == "Hits@1":
        cols = [col for col in data.columns if 'hits@1' in col and 'hits@10' not in col]
    elif metric == "Hits@10":
        cols = [col for col in data.columns if 'hits@10' in col]
    elif metric == "MR":
        cols = [col for col in data.columns if 'MR' in col and 'MRR' not in col]
    elif metric == "MRR":
        cols = [col for col in data.columns if 'MRR' in col]
    data = data[cols]

    for c in cols:
        data[c] = data[c].fillna(0)
    t_data = data.T

    if metric == "MR":
        # assign a very high value to set the appropriate rank of
        # AttrE for D_W_15K_V1 and D_W_15K_V2, since AttrE cannot run on these datasets
        t_data[1][5] = 100000
        t_data[2][5] = 100000
        ranked_t_data = t_data.rank(ascending=1).T
    else:
        ranked_t_data = t_data.rank(ascending=0).T

    col = data.columns[1:]
    for c in cols:
        data[c] = data[c].astype(str) + " (" + ranked_t_data[c].astype(str) + ")"

    # print the ranks
    pd.set_option('display.max_columns', None)
    pd.set_option('display.max_rows', None)
    print(data)

    names = list()
    names_unprocessed = data.columns
    for n in names_unprocessed:
        names.append(n.split(" ")[0])

    avranks = list()
    for i in range(0, 8):
        avranks.append(data[data.columns[i]].str.extract(r"\((.*?)\)")[0:8].astype('float').mean()[0])

    print(avranks)
    cd = Orange.evaluation.compute_CD(avranks, 8, alpha="0.1")
    print("cd=", cd)
    Orange.evaluation.graph_ranks(avranks, names, cd=cd, width=5.5, textspace=1.5)
    # plt.show()
    plt.savefig("./test_nemenyi.png")