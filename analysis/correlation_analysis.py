import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import scipy.stats

feature_list = ["#entity_pairs", "avg. rel. per entity", "avg. attr. per entity", "Sole (%)", "Hyper (%)",
                "Pred. Sim. (%)", "Lit. Sim. (%)", "Ent Names Sim.", "#ents with :descr.",
                "Desc. Sim."]
path = "performances_and_statistics.xlsx"

def aggregate_ents(data):
    l = list()
    for i in range(0, 16, 2):
        l.append(data['#ents'].iloc[i])
    return l


def pred_sim(data):
    l = list()
    for i in range(0, 8, 1):
        l.append(data['Pred. Sim.'].iloc[i])
    return l


def lit_sim(data):
    l = list()
    for i in range(0, 8, 1):
        l.append(data['Lit. Sim.'].iloc[i])
    return l

def aggregate_ratio(data, col_name):
    l = list()
    for i in range(0, 16, 2):
        l.append(((data[col_name].iloc[i] + data[col_name].iloc[i + 1]) / (
                data['rel.'].iloc[i] + data['rel.'].iloc[i + 1])) * 100)
    return l


def ents_have_descr(data, col_name):
    l = list()
    for i in range(0, 16, 2):
        l.append(data[col_name].iloc[i])
    return l

def AVG_avg_rel_per_entity(data):
    l = list()
    for i in range(0, 16, 2):
        l.append((data['avg. rel. per entity'].iloc[i] + data['avg. rel. per entity'].iloc[i + 1]) / 2)
    return l


def AVG_avg_attr_per_entity(data):
    l = list()
    for i in range(0, 16, 2):
        l.append((data['avg. attr. per entity'].iloc[i] + data['avg. attr. per entity'].iloc[i + 1]) / 2)
    return l

def desc_sim_avg(data):
    l = list()
    for i in range(0, 8):
        if data['Desc. Sim. (AVG of AVG)'].iloc[i] == "0.003":
            l.append(0.003)
        else:
            l.append(float(data['Desc. Sim. (AVG of AVG)'].iloc[i]))
    return l


def desc_sim_avg_max(data):
    l = list()
    for i in range(0, 8):
        l.append(float(data['Desc. Sim. (AVG of MAX)'].iloc[i]))
    return l

def ent_pairs(data, col_name):
    l = list()
    for i in range(0, 16, 2):
        l.append(data[col_name].iloc[i] + data[col_name].iloc[i + 1])
    return l

def ent_names_sim_avg_max(data):
    l = list()
    for i in range(0, 8):
        l.append(float(data['Ent Names Sim. (AVG of MAX)'].iloc[i]))
    return l


data = pd.read_excel(path, skiprows=0)

for c in data.columns:
    data[c] = data[c].apply(pd.to_numeric, errors='coerce')


data_metrics = {}
for col in data.columns[0:36]:
    data_metrics[col] = data[col][0:8]

new_data = {
    '#entity_pairs': aggregate_ents(data),
    'avg. rel. per entity': AVG_avg_rel_per_entity(data),
    'avg. attr. per entity': AVG_avg_attr_per_entity(data),
    'Sole (%)': aggregate_ratio(data, "Sole"),
    'Hyper (%)': aggregate_ratio(data, "Hyper "),
    '#ents with :descr.': ents_have_descr(data, "#ents having :descr."),
    'Pred. Sim. (%)': pred_sim(data),
    'Lit. Sim. (%)': lit_sim(data),
    'Desc. Sim.': desc_sim_avg_max(data),
    'Ent Names Sim.': ent_names_sim_avg_max(data),
}

new_dataframe = {}
new_dataframe.update(data_metrics)
new_dataframe.update(new_data)
df = pd.DataFrame(new_dataframe)

correlations = {}
for i in range(0, 36, 1):
    corr_list = list()
    for j in feature_list:
        df1 = df[df.columns[i]].replace(np.nan, 0)
        df2 = df[j].replace(np.nan, 0)
        corr_list.append((scipy.stats.spearmanr(df1, df2)[0], scipy.stats.spearmanr(df1, df2)[1]))

    temp = list()
    for pair in corr_list:
        if pair[1] < 0.05:
            temp.append(pair[0])
        else:
            temp.append(np.nan)
    mul = temp
    if "MR" in df.columns[i] and "MRR" not in df.columns[i]:
        mul = list()
        for el in temp:
            if el is not np.nan:
                mul.append(el * (-1))
            else:
                mul.append(np.nan)
    correlations[df.columns[i]] = mul

corr_df = pd.DataFrame(correlations)
corr_df.index = pd.Index(feature_list)
plt.figure(figsize=(33, 13))
sns.set(font_scale=1.5)
ax = sns.heatmap(corr_df, vmin=-1, vmax=1, annot=True,
                 cmap=sns.cubehelix_palette(50, hue=0.05, rot=0, light=0.9, dark=0, as_cmap=True), linewidths=.8,
                 square=True)
ax.tick_params(top=True, labeltop=True, bottom=False, labelbottom=False, rotation=0)
plt.xticks(rotation=23)
plt.setp(ax.xaxis.get_majorticklabels(), ha='left')
# plt.show()
plt.savefig("./test_correlation.png")
