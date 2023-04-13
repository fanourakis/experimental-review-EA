import pickle
from transformers import BertTokenizer
import logging
from Param import *
import pickle
import numpy as np
import re
import random
logging.getLogger("transformers.tokenization_utils").setLevel(logging.ERROR)



def get_name(ent, dataset):
    ent_name = ent
    if "EN_JA" in dataset:
        if "http://dbpedia.org/resource" in ent:
            ent_name = ent.split("http://dbpedia.org/resource/")[-1]
        else:
            ent_name = ent.split("http://ja.dbpedia.org/resource/")[-1]
    elif "EN_DE" in dataset:
        if "http://dbpedia.org/resource" in ent:
            ent_name = ent.split("http://dbpedia.org/resource/")[-1]
        else:
            ent_name = ent.split("http://de.dbpedia.org/resource/")[-1]
    elif "EN_FR" in dataset:
        if "http://dbpedia.org/resource" in ent:
            ent_name = ent.split("http://dbpedia.org/resource/")[-1]
        else:
            ent_name = ent.split("http://fr.dbpedia.org/resource/")[-1]
    elif "DBP_en_YG_en" in dataset or "D_Y" in dataset:
        if "http://dbpedia.org/resource" in ent:
            ent_name = ent.split("http://dbpedia.org/resource/")[-1]
        else:
            ent_name = ent
    elif "DBP_en_WD_en" in dataset or "D_W" in dataset:
        if "http://dbpedia.org/resource" in ent:
            ent_name = ent.split("http://dbpedia.org/resource/")[-1]
        else:
            ent_name = ent.split("http://www.wikidata.org/entity/")[-1]
    elif "bbc" in dataset:
        if "dbp:" in ent:
            ent_name = ent.split("dbp:")[-1]
        else:
            ent_name = ent.split("/")[-1]
    else:
        ent_name = ent.split("/")[-1]

    ent_name = ent_name.replace('_', ' ')
    return ent_name



def ent2desTokens_generate(Tokenizer,des_dict_path,ent_list_1,ent_list_2,des_limit = DES_LIMIT_LENGTH - 2):
    #ent_list_1/2 == two different language ent list
    print("load desription data from... :", des_dict_path)
    ori_des_dict = pickle.load(open(des_dict_path,"rb"))
    ent2desTokens = dict()
    ent_set_1 = set(ent_list_1)
    ent_set_2 = set(ent_list_2)
    for ent,ori_des in ori_des_dict.items():
        if ent not in ent_set_1 and ent not in ent_set_2:
            continue
        string = ori_des
        encode_indexs = Tokenizer.encode(string)[:des_limit]
        ent2desTokens[ent] = encode_indexs
    print("The num of entity with description:",len(ent2desTokens.keys()))
    return ent2desTokens



def ent2Tokens_gene(Tokenizer,ent2desTokens,ent_list,index2entity,
                                ent_name_max_length = DES_LIMIT_LENGTH - 2):
    ent2tokenids = dict()
    for ent_id in ent_list:
        ent = index2entity[ent_id]
        if ent2desTokens!= None and ent in ent2desTokens:
            #if entity has description, use entity description
            token_ids = ent2desTokens[ent]
            ent2tokenids[ent_id] = token_ids
        else:
            #else, use entity name.
            ent_name = get_name(ent)
            token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
            ent2tokenids[ent_id] = token_ids
    return ent2tokenids



def ent2bert_input(ent_ids,Tokenizer,ent2token_ids,des_max_length=DES_LIMIT_LENGTH):
    ent2data = dict()
    pad_id = Tokenizer.pad_token_id

    for ent_id in ent_ids:
        ent2data[ent_id] = [[],[]]
        ent_token_id = ent2token_ids[ent_id]
        ent_token_ids = Tokenizer.build_inputs_with_special_tokens(ent_token_id)

        token_length = len(ent_token_ids)
        assert token_length <= des_max_length

        ent_token_ids = ent_token_ids + [pad_id] * max(0, des_max_length - token_length)

        ent_mask_ids = np.ones(np.array(ent_token_ids).shape)
        ent_mask_ids[np.array(ent_token_ids) == pad_id] = 0
        ent_mask_ids = ent_mask_ids.tolist()

        ent2data[ent_id][0] = ent_token_ids
        ent2data[ent_id][1] = ent_mask_ids
    return ent2data



def get_poss_attr(attr_path):
    ents_attr = {}
    priority = {}



    if "EN_JA" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://xmlns.com/foaf/0.1/nick": 2, "http://dbpedia.org/ontology/synonym": 3,
                        "http://dbpedia.org/ontology/alias": 4, "http://dbpedia.org/ontology/office": 5,
                        "http://dbpedia.org/ontology/background": 5, "http://dbpedia.org/ontology/leaderTitle": 5,
                        "http://dbpedia.org/ontology/orderInOffice": 5}
        else:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/title": 1,
                        "http://dbpedia.org/ontology/commonName": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://dbpedia.org/ontology/givenName": 4, "http://dbpedia.org/ontology/alias": 5,
                        "http://dbpedia.org/ontology/background": 6, "http://dbpedia.org/ontology/purpose": 6}
    elif "EN_DE" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/title": 1,
                        "http://dbpedia.org/ontology/birthName": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://dbpedia.org/ontology/office": 4, "http://dbpedia.org/ontology/leaderTitle": 5,
                        "http://dbpedia.org/ontology/orderInOffice": 5}
        else:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/originalTitle": 1,
                        "http://xmlns.com/foaf/0.1/nick": 2, "http://dbpedia.org/ontology/motto": 3,
                        "http://dbpedia.org/ontology/leaderTitle": 4}
    elif "EN_FR" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/title": 1,
                        "http://dbpedia.org/ontology/birthName": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://dbpedia.org/ontology/office": 4, "http://dbpedia.org/ontology/leaderTitle": 5,
                        "http://dbpedia.org/ontology/motto": 5, "http://dbpedia.org/ontology/combatant": 5}
        else:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://xmlns.com/foaf/0.1/nick": 2, "http://dbpedia.org/ontology/peopleName": 3,
                        "http://dbpedia.org/ontology/thumbnailCaption": 4, "http://dbpedia.org/ontology/flag": 4,
                        "http://dbpedia.org/ontology/motto": 5, "http://dbpedia.org/ontology/title": 5}
    elif "DBP_en_YG_en" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://xmlns.com/foaf/0.1/nick": 2, "http://dbpedia.org/ontology/alias": 3,
                        "http://dbpedia.org/ontology/office": 4, "http://dbpedia.org/ontology/leaderTitle": 4,
                        "http://dbpedia.org/ontology/motto": 5, "http://dbpedia.org/ontology/combatant": 5}
        else:
            priority = {"skos:prefLabel": 0, "rdfs:label": 1,
                        "redirectedFrom": 2, "hasFamilyName": 3,
                        "hasGivenName": 4, "hasMotto": 5}
    elif "DBP_en_WD_en" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://dbpedia.org/ontology/title": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://dbpedia.org/ontology/synonym": 4, "http://dbpedia.org/ontology/leaderTitle": 4,
                        "http://dbpedia.org/ontology/motto": 5, "http://dbpedia.org/ontology/office": 5}
        else:
            priority = {"http://www.w3.org/2000/01/rdf-schema#label": 0, "http://schema.org/name": 1,
                        "http://www.w3.org/2004/02/skos/core#prefLabel": 2,
                        "http://www.wikidata.org/prop/direct/P373": 3,
                        "http://www.w3.org/2004/02/skos/core#altLabel": 4, "http://schema.org/description": 5,
                        "http://www.wikidata.org/prop/direct/P1549": 6}
    elif "D_W" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://purl.org/dc/elements/1.1/description": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://xmlns.com/foaf/0.1/givenName": 4, "http://dbpedia.org/ontology/leaderTitle": 5,
                        "http://dbpedia.org/ontology/alias": 6,
                        "http://dbpedia.org/ontology/motto": 7, "http://dbpedia.org/ontology/office": 7}
        else:
            priority = {"http://www.wikidata.org/entity/P373": 0, "http://schema.org/description": 1,
                        "http://www.wikidata.org/entity/P1476": 2,
                        "http://www.wikidata.org/entity/P935": 3,
                        "http://www.w3.org/2004/02/skos/core#altLabel": 4}
    elif "D_Y" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://xmlns.com/foaf/0.1/name": 0, "http://dbpedia.org/ontology/birthName": 1,
                        "http://purl.org/dc/elements/1.1/description": 2, "http://xmlns.com/foaf/0.1/nick": 3,
                        "http://xmlns.com/foaf/0.1/givenName": 4, "http://dbpedia.org/ontology/leaderTitle": 5,
                        "http://dbpedia.org/ontology/alias": 6,
                        "http://dbpedia.org/ontology/motto": 7, "http://dbpedia.org/ontology/office": 7}
        else:
            priority = {"skos:prefLabel": 0,
                        "redirectedFrom": 1, "hasFamilyName": 2,
                        "hasGivenName": 3, "hasMotto": 4}
    elif "bbc" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://purl.org/dc/elements/1.1/title": 0, "http://xmlns.com/foaf/0.1/name": 1,
                        "http://open.vocab.org/terms/sortlabel": 2}
        else:
            priority = {"prop:title": 0}
    elif "imdb-tmdb" in attr_path or "imdb-tvdb" in attr_path or "tmdb-tvdb" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"https://www.scads.de/movieBenchmark/ontology/name": 0, "https://www.scads.de/movieBenchmark/ontology/title": 1,
                        "https://www.scads.de/movieBenchmark/ontology/originalTitle": 2}
        else:
            priority = {"https://www.scads.de/movieBenchmark/ontology/name": 0, "https://www.scads.de/movieBenchmark/ontology/title": 1,
                        "https://www.scads.de/movieBenchmark/ontology/originalTitle": 2}
    elif "rest" in attr_path:
        if "attr_triples1" in attr_path:
            priority = {"http://www.okkam.org/ontology_restaurant1.owl#name": 0}
        else:
            priority = {"http://www.okkam.org/ontology_restaurant2.owl#name": 0}
    with open(attr_path) as f:
        for l in f:
            (e, p, a) = l.rstrip("\n").split("\t")     
            if p in priority:
                if e in ents_attr:
                    if priority[p] < priority[ents_attr[e][0]]:
                        ents_attr[e] = (p, a)
                else:
                    ents_attr[e] = (p, a)
    return {e: a for e, (_, a) in ents_attr.items()}


def ent2Tokens_gene_with_attr(Tokenizer, ent2desTokens, ent_list, index2entity, ents_attrs, dataset,
                              ent_name_max_length=DES_LIMIT_LENGTH - 2):
    ent2tokenids = dict()
    for ent_id in ent_list:
        ent = index2entity[ent_id]
        if ent2desTokens != None and ent in ent2desTokens:
            # if entity has description, use entity description
            token_ids = ent2desTokens[ent]
            ent2tokenids[ent_id] = token_ids
        elif ent in ents_attrs:
            # if has a good attribute, use that attribute
            token_ids = Tokenizer.encode(ents_attrs[ent])[:ent_name_max_length]
            ent2tokenids[ent_id] = token_ids
        else:
            # else, use entity name.
            ent_name = get_name(ent, dataset)
            token_ids = Tokenizer.encode(ent_name)[:ent_name_max_length]
            ent2tokenids[ent_id] = token_ids
    return ent2tokenids





def read_data(data_path = DATA_PATH,des_dict_path = DES_DICT_PATH):
    def read_idtuple_file(file_path):
        print('loading a idtuple file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding='utf-8') as f:
            for line in f:
                th = line.strip('\n').split('\t')
                x = []
                for i in range(len(th)):
                    x.append(int(th[i]))
                ret.append(tuple(x))
        return ret
    def read_id2object(file_paths):
        id2object = {}
        for file_path in file_paths:
            with open(file_path, 'r', encoding='utf-8') as f:
                print('loading a (id2object)file...  ' + file_path)
                for line in f:
                    th = line.strip('\n').split('\t')
                    id2object[int(th[0])] = th[1]
        return id2object
    def read_idobj_tuple_file(file_path):
        print('loading a idx_obj file...   ' + file_path)
        ret = []
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                th = line.strip('\n').split('\t')
                ret.append( ( int(th[0]),th[1] ) )
        return ret

    print("load data from... :", data_path)
    #ent_index(ent_id)2entity / relation_index(rel_id)2relation
    index2entity = read_id2object([data_path + "ent_ids_1",data_path + "ent_ids_2"])
    index2rel = read_id2object([data_path + "rel_ids_1",data_path + "rel_ids_2"])
    entity2index = {e:idx for idx,e in index2entity.items()}
    rel2index = {r:idx for idx,r in index2rel.items()}

    #triples
    rel_triples_1 = read_idtuple_file(data_path + 'triples_1')
    rel_triples_2 = read_idtuple_file(data_path + 'triples_2')
    index_with_entity_1 = read_idobj_tuple_file(data_path + 'ent_ids_1')
    index_with_entity_2 = read_idobj_tuple_file(data_path + 'ent_ids_2')

    #ill
    train_ill = read_idtuple_file(data_path + 'sup_pairs')
    test_ill = read_idtuple_file(data_path + 'ref_pairs')
    ent_ill = []
    ent_ill.extend(train_ill)
    ent_ill.extend(test_ill)

    #ent_idx
    entid_1 = [entid for entid,_ in index_with_entity_1]
    entid_2 = [entid for entid,_ in index_with_entity_2]
    entids = list(range(len(index2entity)))

    ents_attr_1 = get_poss_attr(data_path + "attr_triples1")
    ents_attr_2 = get_poss_attr(data_path + "attr_triples2")
    ents_attrs = {}
    ents_attrs.update(ents_attr_1)
    ents_attrs.update(ents_attr_2)

    #ent2descriptionTokens
    Tokenizer = BertTokenizer.from_pretrained('bert-base-multilingual-cased')
    if des_dict_path!= None:
        ent2desTokens = ent2desTokens_generate(Tokenizer,des_dict_path,[index2entity[id] for id in entid_1],[index2entity[id] for id in entid_2])
    else:
        ent2desTokens = None

    #ent2basicBertUnit_input.
    ent2tokenids = ent2Tokens_gene_with_attr(Tokenizer, ent2desTokens, entids, index2entity, ents_attrs, data_path)
    ent2data = ent2bert_input(entids,Tokenizer,ent2tokenids)

    return ent_ill, train_ill, test_ill, index2rel, index2entity, rel2index, entity2index, ent2data, rel_triples_1, rel_triples_2



