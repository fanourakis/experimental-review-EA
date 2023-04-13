"""
hyper-parameters:
"""
CUDA_NUM = 0 #GPU num
LANG = 'ja' #language 'zh'/'ja'/'fr'
ENTITY_NEIGH_MAX_NUM = 50 # max sampling neighbor num of entity
ENTITY_ATTVALUE_MAX_NUM = 50 #max sampling attributeValue num of entity
KERNEL_NUM = 21
SEED_NUM = 11037
CANDIDATE_NUM = 1000 # candidate number

FOLD = "2"
DATASET = "D_W_15K_V2"

BATCH_SIZE = 128 # train batch size
NEG_NUM = 5 # negative sampling num
LEARNING_RATE = 5e-4 # learning rate
MARGIN = 1 # margin
EPOCH_NUM = 200 # train epoch num

INTERACTION_MODEL_SAVE_PATH = "../Save_model/BPR_" + DATASET + "/interaction_model_fold" + FOLD #interaction model save path.

#load model(base_bert_unit_model) path
BASIC_BERT_UNIT_MODEL_SAVE_PATH = "../Save_model/BPR_" + DATASET + "/" 
BASIC_BERT_UNIT_MODEL_SAVE_PREFIX = "fold" + FOLD
LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM = 4
BASIC_BERT_UNIT_MODEL_OUTPUT_DIM = 300

#load data path
DATA_PATH = r"../survey_data/BPR_" + DATASET + "/"


#candidata_save_path
TRAIN_CANDIDATES_PATH = DATA_PATH + "temp/" + FOLD + "/" +'train_candidates.pkl'
TEST_CANDIDATES_PATH = DATA_PATH + "temp/" + FOLD + "/" +'test_candidates.pkl'

#entity embedding and attributeValue embedding save path.
ENT_EMB_PATH = DATA_PATH + '{}_emb_{}.pkl'.format(BASIC_BERT_UNIT_MODEL_SAVE_PREFIX,LOAD_BASIC_BERT_UNIT_MODEL_EPOCH_NUM)
ATTRIBUTEVALUE_EMB_PATH = DATA_PATH + "temp/" + FOLD + "/" +'attributeValue_embedding.pkl'
ATTRIBUTEVALUE_LIST_PATH = DATA_PATH + "temp/" + FOLD + "/" +'attributeValue_list.pkl' #1-1 match to attributeValue embedding.

#(candidate) entity_pairs save path.
ENT_PAIRS_PATH = DATA_PATH + "temp/" + FOLD + "/" +'ent_pairs.pkl' #[(e1,ea),(e1,eb)...]

#interaction feature save filepath name
NEIGHBORVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + "temp/" + FOLD + "/" +'neighbor_view_similarity_feature.pkl' #1-1 match to entity_pairs
ATTRIBUTEVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + "temp/" + FOLD + "/" +'attribute_similarity_feature.pkl' #1-1 match to entity_pairs
DESVIEW_SIMILARITY_FEATURE_PATH = DATA_PATH + "temp/" + FOLD + "/" +'des_view_similarity_feature.pkl' #1-1 match to entity_pairs
