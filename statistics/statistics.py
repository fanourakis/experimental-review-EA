import Levenshtein as lev
import pickle
from sklearn import preprocessing
import numpy.matlib 
import numpy as np
import re
import string

def ent_dict(path, part):
    counter = 0
    triples = {}
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
            if part == "left" or part == "both":
                if line.split("\t", 2)[0] not in triples.keys():
                    triples[line.split("\t", 2)[0]] = list()
                    triples[line.split("\t", 2)[0]].append(
                        (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
                else:
                    triples[line.split("\t", 2)[0]].append(
                        (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
            if part == "right" or part == "both":
                if line.split("\t", 2)[2].rstrip("\n") not in triples.keys():
                    triples[line.split("\t", 2)[2].rstrip("\n")] = list()
                    triples[line.split("\t", 2)[2].rstrip("\n")].append(
                        (line.split("\t", 2)[0], line.split("\t", 2)[1]))
                else:
                    triples[line.split("\t", 2)[2].rstrip("\n")].append(
                        (line.split("\t", 2)[0], line.split("\t", 2)[1]))
    return triples


def ent_dict_attr(path):
    triples = {}
    counter = 0
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
            if line.split("\t", 2)[0] not in triples.keys():
                triples[line.split("\t", 2)[0]] = list()
                triples[line.split("\t", 2)[0]].append(
                    (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
            else:
                triples[line.split("\t", 2)[0]].append(
                    (line.split("\t", 2)[1], line.split("\t", 2)[2].rstrip("\n")))
    return triples


def num_of_triples(path):
    counter = 0
    with open(path, "r") as fp:
        for line in fp:
            counter += 1
    return counter

def pred_sim_rel(dataset_name):
    with open(path + "/rel_triples_1") as fp:
        pred1 = set()
        for line in fp:
            pr = line.split("\t")[1].replace("<", "").replace(">", "")
            if "#" in pr:
                pred1.add(pr.rsplit("#")[-1])
            elif "/" in pr:
                pred1.add(pr.rsplit("/")[-1])
            elif ":" in pr:
                pred1.add(pr.rsplit(":")[-1])
            else:
                pred1.add(pr)
    with open(path + "/rel_triples_2") as fp:
        pred2 = set()
        for line in fp:
            pr = line.split("\t")[1].replace("<", "").replace(">", "")
            if "#" in pr:
                pred2.add(pr.rsplit("#")[-1])
            elif "/" in pr:
                pred2.add(pr.rsplit("/")[-1])
            elif ":" in pr:
                pred2.add(pr.rsplit(":")[-1])
            else:
                pred2.add(pr)

    if "D_W" in dataset_name:
        return 0

    final_sum = 0
    for p1 in pred1:
        sum = 0
        max1 = 0
        for p2 in pred2:
            sum = lev.ratio(p1, p2)
            if sum > max1:
                max1 = sum
        final_sum += max1
    rel_avg = final_sum / len(pred1)
    return rel_avg

def pred_sim_attr(dataset_name):
    with open(path + "/attr_triples_1") as fp:
        pred1 = set()
        for line in fp:
            pr = line.split("\t")[1].replace("<", "").replace(">", "")
            if "#" in pr:
                pred1.add(pr.rsplit("#")[-1])
            elif "/" in pr:
                pred1.add(pr.rsplit("/")[-1])
            elif ":" in pr:
                pred1.add(pr.rsplit(":")[-1])
            else:
                pred1.add(pr)
    with open(path + "/attr_triples_2") as fp:
        pred2 = set()
        for line in fp:
            pr = line.split("\t")[1].replace("<", "").replace(">", "")
            if "#" in pr:
                pred2.add(pr.rsplit("#")[-1])
            elif "/" in pr:
                pred2.add(pr.rsplit("/")[-1])
            elif ":" in pr:
                pred2.add(pr.rsplit(":")[-1])
            else:
                pred2.add(pr)

    if "D_W" in dataset_name:
        return 0

    final_sum = 0
    for p1 in pred1:
        sum = 0
        max2 = 0
        for p2 in pred2:
            sum = lev.ratio(p1, p2)
            if sum > max2:
                max2 = sum
        final_sum += max2
    attr_avg = final_sum / len(pred1)
    return attr_avg

def pred_sim(dataset_name):
    print("Predicates Similarity")
    total = (pred_sim_rel(dataset_name) + pred_sim_attr(dataset_name)) / 2
    print(total)    

def entity_pairs(path):
    ent_dict = {}
    with open(path + "ent_links", "r") as fp:
        for line in fp:
            ent_dict[line.split("\t")[0]] = line.split("\t")[1].rstrip()
    return len(ent_dict.keys())

def entity_name_sim(name):
    if(name == "D_W_15K_V1" or name == "D_W_15K_V2"):
        ents1 = set()
        ents2 = set()
        preds = ['http://www.wikidata.org/entity/P373', 'http://www.wikidata.org/entity/P1476']
        with open("datasets/" + name +"/attr_triples_1", "r") as fp:
            for line in fp:
                if "name" in line.split("\t")[1]:
                    ents1.add(line.split("\t")[2].rstrip())
        with open("datasets/" + name +"/attr_triples_2", "r") as fp:
            for line in fp:
                if line.split("\t")[1] in preds:
                    ents2.add(line.split("\t")[2].rstrip())
    else:
        ents1 = set()
        ents2 = set()
        with open("datasets/" + name +"/721_5fold/1/test_links", "r") as fp:
            for line in fp:
                ents1.add(line.split("\t")[0].split("/")[-1])
                ents2.add(line.split("\t")[1].rstrip("\n").split("/")[-1])
        with open("datasets/" + name +"/721_5fold/1/valid_links", "r") as fp:
            for line in fp:
                ents1.add(line.split("\t")[0].split("/")[-1])
                ents2.add(line.split("\t")[1].rstrip("\n").split("/")[-1])
        with open("datasets/" + name +"/721_5fold/1/train_links", "r") as fp:
            for line in fp:
                ents1.add(line.split("\t")[0].split("/")[-1])
                ents2.add(line.split("\t")[1].rstrip("\n").split("/")[-1])

    ent_dict_1 = {}
    for e1 in ents1:
        max_num = 0
        ent_dict_1[e1] = 0
        for e2 in ents2:
            sim = lev.ratio(e1,e2)
            if sim > max_num:
                max_num = sim
        ent_dict_1[e1] = max_num 
    final_sum = 0
    for key in ent_dict_1.keys():
        final_sum += ent_dict_1[key]
    print(str(final_sum / len(ent_dict_1)))

def ents_with_descriptions(path, kg, dataset):
    ents = set()
    pred_set = set()
    with open(path, "r") as fp:
        for line in fp:
            ent1 = line.split("\t")[0]
            pred = line.split("\t")[1]
            if "name" in pred:
                if kg == 1:
                    if ent1 in valid_ents1:
                        ents.add(ent1)
                elif kg == 2:
                    if ent1 in valid_ents2:
                        ents.add(ent1)
                pred_set.add(pred)
            if dataset in ["D_W_15K_V1", "D_W_15K_V2"]:
                if pred in ["http://www.wikidata.org/entity/P373", "http://www.wikidata.org/entity/P1476"]:
                    pred_set.add(pred)
                    if kg == 1:
                        if ent1 in valid_ents1:
                            ents.add(ent1)
                    elif kg == 2:
                        if ent1 in valid_ents2:
                            ents.add(ent1)
            elif dataset in ["D_Y_15K_V1", "D_Y_15K_V2"]:
                if pred in ["http://dbpedia.org/ontology/birthName", "skos:prefLabel"]:
                    pred_set.add(pred)
                    if kg == 1:
                        if ent1 in valid_ents1:
                            ents.add(ent1)
                    elif kg == 2:
                        if ent1 in valid_ents2:
                            ents.add(ent1)
  
    return len(ents)


def get_ents(valid_path, test_path):
    valid_ents1 = set()
    valid_ents2 = set()
    with open(valid_path, "r") as fp:
        for line in fp:
            ent1 = line.split("\t")[0]
            ent2 = line.split("\t")[1].rstrip("\n")
            valid_ents1.add(ent1)
            valid_ents2.add(ent2)
    with open(test_path, "r") as fp:
        for line in fp:
            ent1 = line.split("\t")[0]
            ent2 = line.split("\t")[1].rstrip("\n")
            valid_ents1.add(ent1)
            valid_ents2.add(ent2)
    return valid_ents1, valid_ents2

def descr_sim(dataset_name):
    path = "descriptions_pickles/" + dataset_name + "/"
    with open(path + "emb1_" + dataset_name + ".pickle", 'rb') as f1:
        emb1 = pickle.load(f1)
    with open(path + "emb2_" + dataset_name + ".pickle", 'rb') as f2:
        emb2 = pickle.load(f2)

    a1,b1,c1 = emb1.shape
    emb1 = emb1.reshape((a1,b1*c1))
    a2,b2,c2 = emb2.shape
    emb2 = emb2.reshape((a2,b2*c2))

    emb1_norm = preprocessing.normalize(emb1)
    emb2_norm = preprocessing.normalize(emb2)

    mul = np.matmul(emb1_norm, emb2_norm.T)
    s = 0
    counter = 0
    for i in mul:
        s += max(i)
        counter += 1

    return s/counter

def is_number(s):
    try:
        float(s)
        return True
    except ValueError:
        pass
    try:
        import unicodedata
        unicodedata.numeric(s)
        return True
    except (TypeError, ValueError):
        pass
    return False


def clear_attribute_triples(attribute_triples):
    print('\nbefore clear:', len(attribute_triples))
    # step 1
    attribute_triples_new = set()
    attr_num = {}
    for (e, a, _) in attribute_triples:
        ent_num = 1
        if a in attr_num:
            ent_num += attr_num[a]
        attr_num[a] = ent_num
    attr_set = set(attr_num.keys())
    attr_set_new = set()
    for a in attr_set:
        if attr_num[a] >= 10:
            attr_set_new.add(a)
    for (e, a, v) in attribute_triples:
        if a in attr_set_new:
            attribute_triples_new.add((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 1:', len(attribute_triples))

    # step 2
    attribute_triples_new = []
    literals_number, literals_string = [], []
    for (e, a, v) in attribute_triples:
        v = v.strip('"')
        if '"^^' in v:
            v = v[:v.index('"^^')]
        if v.endswith('"@en'):
            v = v[:v.index('"@en')]
        if v.endswith('"@eng'):
            v = v[:v.index('"@eng')]
        if is_number(v):
            literals_number.append(v)
        else:
            literals_string.append(v)
        v = v.replace('.', '').replace('(', '').replace(')', '').replace(',', '').replace('"', '')
        v = v.replace('_', ' ').replace('-', ' ').replace('/', ' ')
        if 'http' in v:
            continue

        attribute_triples_new.append((e, a, v))
    attribute_triples = attribute_triples_new
    print('after step 2:', len(attribute_triples))
    return attribute_triples, literals_number, literals_string

def clean_string(s):
    # return re.sub(r'[^\w\s]', '', s).lower().split(" ")
    return re.sub(r'[^\w\s]'," ",s)

def lit_sim_avg(cattr1, cattr2):
    attr_dict = {}
    for v1 in cattr1:
        counter = 0
        attr_dict[v1] = 0
        max1 = 0 
        for v2 in cattr2:
            sim = lev.ratio(v1, v2)
            if sim > max1:
                max1 = sim
        attr_dict[v1] = max1

    final_sum = 0
    for key in attr_dict.keys():
        final_sum += attr_dict[key]
    print(str(final_sum / len(attr_dict.keys())))

def graph_to_dict(path, part, kg):
    triples = {}
    path = path + "rel_triples_" + kg
    with open(path, "r") as fp:
        for line in fp:
            if part == "pair":
                if (line.split("\t", 2)[0], line.split("\t", 2)[2].rstrip("\n")) not in triples.keys():
                    triples[(line.split("\t", 2)[0], line.split("\t", 2)[2].rstrip("\n"))] = list()
                    triples[(line.split("\t", 2)[0], line.split("\t", 2)[2].rstrip("\n"))].append(
                        line.split("\t", 2)[1])
                else:
                    triples[(line.split("\t", 2)[0], line.split("\t", 2)[2].rstrip("\n"))].append(
                        line.split("\t", 2)[1])
    return triples

def sole_relations(path, dict, kg):
    path = path + "rel_triples_" + kg
    relations = set()
    with open(path, "r") as fp:
        for line in fp:
            relations.add(line.split("\t", 2)[1])

    non_sole = set()
    for r in relations:
        for i in dict.keys():
            if r in dict[i] and len(set(dict[i])) > 1:
                non_sole.add(r)
                break
    
    return len(relations) - len(non_sole), len(relations)


def hyper_relations(path, dict, kg):
    path = path + "rel_triples_" + kg
    hyper_relations_set = set()
    relations = set()
    with open(path, "r") as fp:
        for line in fp:
            relations.add(line.split("\t", 2)[1])
    for r in relations:
        for i in dict.keys():
            if r in dict[i] and len(set(dict[i])) > 1:
                for rel in dict[i]:
                    hyper_relations_set.add(rel)
    return len(hyper_relations_set), len(relations)


name = "D_Y_15K_V1"
path = "datasets/" + name +"/"

num_of_rel_triples_1 = num_of_triples(path + "/rel_triples_1")
num_of_rel_triples_2 = num_of_triples(path + "/rel_triples_2")

num_of_attr_triples_1 = num_of_triples(path + "/attr_triples_1")
num_of_attr_triples_2 = num_of_triples(path + "/attr_triples_2")

ent_dict_1 = ent_dict(path + "/rel_triples_1", "both")
ent_dict_2 = ent_dict(path + "/rel_triples_2", "both")

ent_dict_attr_1 = ent_dict_attr(path + "/attr_triples_1")
ent_dict_attr_2 = ent_dict_attr(path + "/attr_triples_2")

ent_dict_left_1 = ent_dict(path + "/rel_triples_1", "left")
ent_dict_left_2 = ent_dict(path + "/rel_triples_2", "left")

ent_dict_attr_left_1 = ent_dict(path + "/attr_triples_1", "left")
ent_dict_attr_left_2 = ent_dict(path + "/attr_triples_2", "left")

# # average relations per entity.
avg1_rel = num_of_rel_triples_1/len(ent_dict_left_1.keys())
avg2_rel = num_of_rel_triples_2/len(ent_dict_left_2.keys())
print("------Average relations per entity---------")
print(avg1_rel)
print(avg2_rel)

# # average attributes per entity
avg1_attr = num_of_attr_triples_1/len(ent_dict_attr_left_1.keys())
avg2_attr = num_of_attr_triples_2/len(ent_dict_attr_left_2.keys())
print("------Average attributes per entity---------")
print(avg1_attr)
print(avg2_attr)


# # Seed Alignment Size
print("Seed Alignment Size")
print(entity_pairs(path))

# # Predicates Similarity
pred_sim(path)

# # Entity Names Similarity
print("Entity names similarity")
entity_name_sim(name)

# #Ents that hae descriptions
valid_path = path + "721_5fold/1/valid_links"
test_path = path + "721_5fold/1/test_links"
valid_ents1, valid_ents2 = get_ents(valid_path, test_path)
print("Entities that have descriptions")
print(ents_with_descriptions(path + "attr_triples_1", 1, name))

# # Description similarity
print("Description Similarity")
print(descr_sim(name))

# Literal Similarity
print("Literal Similarity")
attr1 = set()
with open("datasets/" + name +"/attr_triples_1", "r") as fp:
    for line in fp:
        if "^^" in line.split("\t", 2)[2].rstrip("\n"):
            attr1.add(line.split("\t", 2)[2].rstrip("\n").split("^^")[0].replace("\"", ""))
        else:
            attr1.add(line.split("\t", 2)[2].rstrip("\n"))
attr1_cleaned = set()
for a1 in attr1:
    attr1_cleaned.add(clean_string(a1))


attr2 = set()
with open("datasets/" + name +"/attr_triples_2", "r") as fp:
    for line in fp:
        if "^^" in line.split("\t", 2)[2].rstrip("\n"):
            attr2.add(line.split("\t", 2)[2].rstrip("\n").split("^^")[0].replace("\"", ""))
        else:
            attr2.add(line.split("\t", 2)[2].rstrip("\n"))
attr2_cleaned = set()
for a2 in attr2:
    attr2_cleaned.add(clean_string(a2))

lit_sim_avg(attr1_cleaned, attr2_cleaned)

# Sole and Hyper
print("Sole")
pair_part_dict_1 = graph_to_dict(path, "pair", "1")
pair_part_dict_2 = graph_to_dict(path, "pair", "2")
soles_1 , rels1 = sole_relations(path, pair_part_dict_1, "1")
soles_2 , rels2 = sole_relations(path, pair_part_dict_2, "2")
print(soles_1)
print(soles_2)

print("Hyper")
hyper_1 , rels1 = hyper_relations(path, pair_part_dict_1, "1")
hyper_2 , rels2 = hyper_relations(path, pair_part_dict_2, "2")
print(hyper_1)
print(hyper_2)