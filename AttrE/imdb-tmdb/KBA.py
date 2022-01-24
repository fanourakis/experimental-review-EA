
# coding: utf-8

# In[ ]:


from rdflib import Graph, URIRef
import random
import numpy as np
import tensorflow as tf
import math
import datetime as dt
import cPickle
import rdflib
import re
import collections
from tensorflow.contrib import rnn

import os
DEVICE = "0"
os.environ["CUDA_VISIBLE_DEVICES"] = DEVICE


# #### Load Data

# In[ ]:


def invert_dict(d):
    return dict([(v, k) for k, v in d.iteritems()])

# KB1 & KB2 entity and literal vocab
entity_literal_vocab = cPickle.load(open("data/vocab_all.pickle", "rb"))
# KB1 & KB2 character vocab
char_vocab = cPickle.load(open("data/vocab_char.pickle", "rb"))
entity_vocab = cPickle.load(
    open("data/vocab_entity.pickle", "rb"))  # KB1 & KB2 entity vocab
predicate_vocab = cPickle.load(
    open("data/vocab_predicate.pickle", "rb"))  # KB1 & KB2 predicate vocab
# KB1 entity vocab for filtering final result
entity_kb1_vocab = cPickle.load(open("data/vocab_kb1.pickle", "rb"))
# KB1 entity & literal vocab for negative sampling
entity_kb1_vocab_neg = cPickle.load(open("data/vocab_kb1_neg.pickle", "rb"))
# KB2 entity & literal vocab for negative sampling
entity_kb2_vocab_neg = cPickle.load(open("data/vocab_kb2_neg.pickle", "rb"))
entity_label_dict = cPickle.load(
    open("data/entity_label.pickle", "rb"))  # KB1 & KB2 entity label
entity_literal_kb1_vocab_neg = cPickle.load(
    open("data/vocab_kb1_all_neg.pickle", "rb"))  # KB1 entity & literal vocab
entity_literal_kb2_vocab_neg = cPickle.load(
    open("data/vocab_kb2_all_neg.pickle", "rb"))  # KB1 entity & literal vocab

reverse_entity_vocab = invert_dict(entity_vocab)
reverse_predicate_vocab = invert_dict(predicate_vocab)
reverse_char_vocab = invert_dict(char_vocab)
reverse_entity_literal_vocab = invert_dict(entity_literal_vocab)

# relationship triples & attribute triples
data_uri = cPickle.load(open("data/data_uri.pickle", "rb"))
data_uri_n = cPickle.load(open("data/data_uri_n.pickle", "rb"))
data_literal = cPickle.load(open("data/data_literal.pickle", "rb"))
data_literal_n = cPickle.load(open("data/data_literal_n.pickle", "rb"))


# #### Methods for data processing

# In[ ]:


def dataType(string):
    odp = 'string'
    patternBIT = re.compile('[01]')
    patternINT = re.compile('[0-9]+')
    patternFLOAT = re.compile('[0-9]+\.[0-9]+')
    patternTEXT = re.compile('[a-zA-Z0-9]+')
    if patternTEXT.match(string):
        odp = "string"
    if patternINT.match(string):
        odp = "integer"
    if patternFLOAT.match(string):
        odp = "float"
    return odp


def getRDFData(o):
    if isinstance(o, rdflib.term.URIRef):
        data_type = "uri"
    else:
        data_type = o.datatype
        if data_type == None:
            data_type = dataType(o)
        else:
            if "#" in o.datatype:
                data_type = o.datatype.split('#')[1].lower()
            else:
                data_type = dataType(o)
        if data_type == 'gmonthday' or data_type == 'gyear':
            data_type = 'date'
        if data_type == 'positiveinteger' or data_type == 'int' or data_type == 'nonnegativeinteger':
            data_type = 'integer'
    return o, data_type


def invert_dict(d):
    return dict([(v, k) for k, v in d.iteritems()])


def getLiteralArray(o, literal_len, char_vocab):
    literal_object = list()
    for i in range(literal_len):
        literal_object.append(0)
    if o[1] != 'uri':
        max_len = min(literal_len, len(o[0]))
        for i in range(max_len):
            if char_vocab.get(o[0][i]) == None:
                char_vocab[o[0][i]] = len(char_vocab)
            literal_object[i] = char_vocab[o[0][i]]
    elif entity_label_dict.get(o[0]) != None:
        label = entity_label_dict.get(o[0])
        max_len = min(literal_len, len(label))
        for i in range(max_len):
            if char_vocab.get(label[i]) == None:
                char_vocab[label[i]] = len(char_vocab)
            literal_object[i] = char_vocab[label[i]]
    return literal_object


def getBatch(data, batchSize, current, entityVocab, literal_len, char_vocab):
    hasNext = current + batchSize < len(data)

    if (len(data) - current) < batchSize:
        current = current - (batchSize - (len(data) - current))

    dataPos_all = data[current:current + batchSize]
    dataPos = list()
    charPos = list()
    pred_weight_pos = list()
    dataNeg = list()
    charNeg = list()
    pred_weight_neg = list()
    for triples, chars, pred_weight in dataPos_all:
        s, p, o, p_trans = triples
        dataPos.append([s, p, o, []])
        charPos.append(chars)
        pred_weight_pos.append(pred_weight)
        lr = round(random.random())
        if lr == 0:
            try:
                o_type = getRDFData(reverse_entity_vocab[o])
            except:
                o_type = 'not_uri'

            literal_array = []
            rerun = True
            while rerun or negElm[0] == (reverse_entity_vocab[o] and literal_array == chars):
                if o_type[1] == 'uri':
                    if str(s).startswith('https://www.scads.de/movieBenchmark/resource/IMDB'):
                        negElm = entity_kb1_vocab_neg[
                            random.randint(0, len(entity_kb1_vocab_neg) - 1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                    else:
                        negElm = entity_kb2_vocab_neg[
                            random.randint(0, len(entity_kb2_vocab_neg) - 1)]
                        negElm = reverse_entity_vocab[entity_vocab[negElm]]
                else:
                    if str(s).startswith('https://www.scads.de/movieBenchmark/resource/IMDB'):
                        negElm = entity_literal_kb1_vocab_neg[
                            random.randint(0, len(entity_literal_kb1_vocab_neg) - 1)]
                        negElm = reverse_entity_literal_vocab[
                            entity_literal_vocab[negElm]]
                    else:
                        negElm = entity_literal_kb2_vocab_neg[
                            random.randint(0, len(entity_literal_kb2_vocab_neg) - 1)]
                        negElm = reverse_entity_literal_vocab[
                            entity_literal_vocab[negElm]]
                if o_type == 'uri' and negElm[1] == 'uri':
                    rerun = False
                elif o_type != 'uri':
                    rerun = False
                if (isinstance(negElm, rdflib.term.URIRef)) or (isinstance(negElm, rdflib.term.Literal)):
                    negElm = getRDFData(negElm)
                    literal_array = getLiteralArray(
                        negElm, literal_len, char_vocab)
                else:
                    rerun = True
            if negElm[1] == 'uri':
                dataNeg.append([s, p, entity_vocab[negElm[0]], []])
            else:
                dataNeg.append([s, p, entity_vocab[negElm[1]], []])
            charNeg.append(literal_array)
            pred_weight_neg.append(pred_weight)
        else:
            negElm = random.randint(0, len(entity_vocab) - 1)
            while negElm == s:
                negElm = random.randint(0, len(entity_vocab) - 1)
            dataNeg.append([negElm, p, o, []])
            charNeg.append(chars)
            pred_weight_neg.append(pred_weight)

    dataPos = np.array(dataPos)
    charPos = np.array(charPos)
    pred_weight_pos = np.array(pred_weight_pos)
    dataNeg = np.array(dataNeg)
    charNeg = np.array(charNeg)
    pred_weight_neg = np.array(pred_weight_neg)
    return hasNext, current + batchSize, dataPos[:, 0], dataPos[:, 1], dataPos[:, 2], dataPos[:, 3], pred_weight_pos, charPos, dataNeg[:, 0], dataNeg[:, 1], dataNeg[:, 2], dataNeg[:, 3], pred_weight_neg, charNeg


# #### Hyperparameter

# In[ ]:


batchSize = 100
hidden_size = 100
totalEpoch = 50
verbose = 1000
margin = 1.0
literal_len = 10
entitySize = len(entity_vocab)
predSize = len(predicate_vocab)
charSize = len(char_vocab)
top_k = 10


# #### Prepare testing data

# In[ ]:


import random
from rdflib import URIRef

file_mapping = open("data/ent_links.ttl", 'r')

test_dataset_list = list()
for line in file_mapping:
    elements = line.split(' ')
    s = elements[0]
    p = elements[1]
    o = elements[2]

    if (entity_vocab[URIRef(s.replace('<', '').replace('>', ''))] in entity_kb1_vocab) and (URIRef(o.replace('<', '').replace('>', '')) in entity_vocab):
        test_dataset_list.append((o, s))
file_mapping.close()

test_input = [entity_vocab[
    URIRef(k.replace('<', '').replace('>', ''))] for k, _ in test_dataset_list]
test_answer = [entity_kb1_vocab.index(entity_vocab[URIRef(
    k.replace('<', '').replace('>', ''))]) for _, k in test_dataset_list]


# #### Embedding model

# In[ ]:


tfgraph = tf.Graph()

with tfgraph.as_default():
    pos_h = tf.placeholder(tf.int32, [None])
    pos_t = tf.placeholder(tf.int32, [None])
    pos_r = tf.placeholder(tf.int32, [None])
    pos_c = tf.placeholder(tf.int32, [None, literal_len])
    pos_pred_weight = tf.placeholder(
        tf.float32, [None, 1], name='pos_pred_weight')

    neg_h = tf.placeholder(tf.int32, [None])
    neg_t = tf.placeholder(tf.int32, [None])
    neg_r = tf.placeholder(tf.int32, [None])
    neg_c = tf.placeholder(tf.int32, [None, literal_len])
    neg_pred_weight = tf.placeholder(
        tf.float32, [None, 1], name='neg_pred_weight')

    type_data = tf.placeholder(tf.int32, [1])

    ent_embeddings_ori = tf.get_variable(name="relationship_ent_embedding", shape=[
                                         entitySize, hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    atr_embeddings_ori = tf.get_variable(name="attribute_ent_embedding", shape=[
                                         entitySize, hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    rel_embeddings = tf.get_variable(name="rel_embedding", shape=[
                                     predSize, hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    attribute_rel_embeddings = tf.get_variable(name="attribute_rel_embedding", shape=[
                                               predSize, hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))
    char_embeddings = tf.get_variable(name="attribute_char_embedding", shape=[
                                      charSize, hidden_size], initializer=tf.contrib.layers.xavier_initializer(uniform=False))

    ent_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    ent_indices = tf.reshape(ent_indices, [-1, 1])
    ent_value = tf.concat([tf.nn.embedding_lookup(ent_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, pos_t),
                           tf.nn.embedding_lookup(ent_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(ent_embeddings_ori, neg_t)], 0)
    part_ent_embeddings = tf.scatter_nd(
        [ent_indices], [ent_value], ent_embeddings_ori.shape)
    ent_embeddings = part_ent_embeddings + \
        tf.stop_gradient(-part_ent_embeddings + ent_embeddings_ori)

    atr_indices = tf.concat([pos_h, pos_t, neg_h, neg_t], 0)
    atr_indices = tf.reshape(atr_indices, [-1, 1])
    atr_value = tf.concat([tf.nn.embedding_lookup(atr_embeddings_ori, pos_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, pos_t),
                           tf.nn.embedding_lookup(atr_embeddings_ori, neg_h),                          tf.nn.embedding_lookup(atr_embeddings_ori, neg_t)], 0)
    part_atr_embeddings = tf.scatter_nd(
        [atr_indices], [atr_value], atr_embeddings_ori.shape)
    atr_embeddings = part_atr_embeddings + \
        tf.stop_gradient(-part_atr_embeddings + atr_embeddings_ori)

    pos_h_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        ent_embeddings, pos_h), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_h))
    pos_t_e = tf.cond(type_data[0] > 0, lambda: tf.stop_gradient(tf.nn.embedding_lookup(
        ent_embeddings, pos_t)), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_t))
    pos_r_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        rel_embeddings, pos_r), lambda: tf.nn.embedding_lookup(attribute_rel_embeddings, pos_r))
    pos_c_e = tf.nn.embedding_lookup(char_embeddings, pos_c)
    neg_h_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        ent_embeddings, neg_h), lambda: tf.nn.embedding_lookup(atr_embeddings, neg_h))
    neg_t_e = tf.cond(type_data[0] > 0, lambda: tf.stop_gradient(tf.nn.embedding_lookup(
        ent_embeddings, neg_t)), lambda: tf.nn.embedding_lookup(atr_embeddings, neg_t))
    neg_r_e = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        rel_embeddings, neg_r), lambda: tf.nn.embedding_lookup(attribute_rel_embeddings, neg_r))
    neg_c_e = tf.nn.embedding_lookup(char_embeddings, neg_c)

    mask_constant_0 = np.zeros([1, hidden_size])
    mask_constant_1 = np.ones([1, hidden_size])
    mask_constant = np.concatenate([mask_constant_0, mask_constant_1])
    mask_constant = tf.constant(mask_constant, tf.float32)

    flag_pos_c_e = tf.sign(tf.abs(pos_c))
    mask_pos_c_e = tf.nn.embedding_lookup(mask_constant, flag_pos_c_e)
    pos_c_e = pos_c_e * mask_pos_c_e

    flag_neg_c_e = tf.sign(tf.abs(neg_c))
    mask_neg_c_e = tf.nn.embedding_lookup(mask_constant, flag_neg_c_e)
    neg_c_e = neg_c_e * mask_neg_c_e

    def calculate_ngram_weight(unstacked_tensor):
        stacked_tensor = tf.stack(unstacked_tensor, 1)
        stacked_tensor = tf.reverse(stacked_tensor, [1])
        index = tf.constant(len(unstacked_tensor))
        expected_result = tf.zeros([batchSize, hidden_size])

        def condition(index, summation):
            return tf.greater(index, 0)

        def body(index, summation):
            precessed = tf.slice(
                stacked_tensor, [0, index - 1, 0], [-1, -1, -1])
            summand = tf.reduce_mean(precessed, 1)
            return tf.subtract(index, 1), tf.add(summation, summand)
        result = tf.while_loop(condition, body, [index, expected_result])
        return result[1]

    pos_c_e_in_lstm = tf.unstack(pos_c_e, literal_len, 1)
    pos_c_e_lstm = calculate_ngram_weight(pos_c_e_in_lstm)

    neg_c_e_in_lstm = tf.unstack(neg_c_e, literal_len, 1)
    neg_c_e_lstm = calculate_ngram_weight(neg_c_e_in_lstm)

    tail_pos = tf.cond(type_data[0] > 0, lambda: pos_t_e, lambda: pos_c_e_lstm)
    tail_neg = tf.cond(type_data[0] > 0, lambda: neg_t_e, lambda: neg_c_e_lstm)

    pos = tf.reduce_sum(abs(pos_h_e + pos_r_e - tail_pos), 1, keep_dims=True)
    neg = tf.reduce_sum(abs(neg_h_e + neg_r_e - tail_neg), 1, keep_dims=True)

    pos = tf.cond(type_data[0] > 0, lambda: pos,
                  lambda: tf.multiply(pos, pos_pred_weight))
    neg = tf.cond(type_data[0] > 0, lambda: neg,
                  lambda: tf.multiply(neg, neg_pred_weight))
    learning_rate = tf.cond(
        type_data[0] > 0, lambda: 0.01, lambda: tf.reduce_min(pos_pred_weight) * 0.01)

    opt_vars_ent = [v for v in tf.trainable_variables() if v.name.startswith(
        "relationship") or v.name.startswith("rel_embedding")]
    opt_vars_atr = [v for v in tf.trainable_variables() if v.name.startswith(
        "attribute") or v.name.startswith("attribute_rel_embedding") or v.name.startswith("rnn")]
    opt_vars_sim = [v for v in tf.trainable_variables() if v.name.startswith(
        "relationship_ent_embedding") or v.name.startswith("attribute_rel_embedding")]
    opt_vars = [v for v in tf.trainable_variables()]

    ent_emb = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        ent_embeddings, pos_t), lambda: tf.nn.embedding_lookup(ent_embeddings, pos_h))
    atr_emb = tf.cond(type_data[0] > 0, lambda: tf.nn.embedding_lookup(
        atr_embeddings, pos_t), lambda: tf.nn.embedding_lookup(atr_embeddings, pos_h))
    norm_ent_emb = tf.nn.l2_normalize(ent_emb, 1)
    norm_atr_emb = tf.nn.l2_normalize(atr_emb, 1)
    cos_sim = tf.reduce_sum(tf.multiply(
        norm_ent_emb, norm_atr_emb), 1, keep_dims=True)
    sim_loss = tf.reduce_sum(1 - cos_sim)
    sim_optimizer = tf.train.AdamOptimizer(
        0.01).minimize(sim_loss, var_list=opt_vars_sim)

    loss = tf.cond(type_data[0] > 0, lambda: tf.reduce_sum(tf.maximum(
        pos - neg + 1, 0) + (1 - cos_sim)), lambda: tf.reduce_sum(tf.maximum(pos - neg + 1, 0)))

    optimizer = tf.cond(type_data[0] > 0, lambda: tf.train.AdamOptimizer(learning_rate).minimize(
        loss, var_list=opt_vars_ent), lambda: tf.train.AdamOptimizer(learning_rate).minimize(loss, var_list=opt_vars_atr))

    norm = tf.sqrt(tf.reduce_sum(
        tf.square(ent_embeddings_ori), 1, keep_dims=True))
    normalized_embeddings = ent_embeddings_ori / norm

    test_dataset = tf.constant(test_input, dtype=tf.int32)
    test_embeddings = tf.nn.embedding_lookup(
        normalized_embeddings, test_dataset)
    similarity = tf.matmul(
        test_embeddings, normalized_embeddings, transpose_b=True)

    init = tf.global_variables_initializer()


# In[ ]:


def metric(y_true, y_pred, answer_vocab, k=10):
    list_rank = list()
    total_hits = 0
    total_hits_1 = 0
    for i in range(len(y_true)):
        result = y_pred[i]
        result = result[answer_vocab]
        result = (-result).argsort()

        for j in range(len(result)):
            if result[j] == y_true[i]:
                rank = j
                break
        list_rank.append(j)
        result = result[:k]
        for j in range(len(result)):
            if result[j] == y_true[i]:
                total_hits += 1
                if j == 0:
                    total_hits_1 += 1
                break
    MR = 0
    MRR = 0
    for a in list_rank:
        incr = (a+1)
        MR += float(incr)
        MRR += float(1) / incr
    MR = MR/len(list_rank)
    MRR = MRR/len(list_rank)
    
    return  MRR, MR, float(total_hits) / len(y_true), float(total_hits_1) / len(y_true)


# In[ ]:


def run(writer, graph, totalEpoch):
    with tf.Session(graph=graph) as session:
        init.run()

        for epoch in range(totalEpoch):
            if epoch % 2 == 0:
                data = [data_uri_n, data_uri, data_literal_n,
                        data_literal, [], []]
            else:
                data = [[], [], data_literal_n, data_literal, [], []]
            start_time_epoch = dt.datetime.now()
            for i in range(0, len(data)):
                random.shuffle(data[i])
                hasNext = True
                current = 0
                step = 0
                average_loss = 0

                if i == 0 or i == 1 or i == 4:
                    uri = 1
                else:
                    uri = 0

                while(hasNext and len(data[i]) > 0):
                    step += 1
                    hasNext, current, ph, pr, pt, pr_trans, ppred, pc, nh, nr, nt, nr_trans, npred, nc = getBatch(
                        data[i], batchSize, current, entity_vocab, literal_len, char_vocab)
                    feed_dict = {
                        pos_h: ph,
                        pos_t: pt,
                        pos_r: pr,
                        pos_pred_weight: ppred,
                        pos_c: pc,
                        neg_h: nh,
                        neg_t: nt,
                        neg_r: nr,
                        neg_c: nc,
                        neg_pred_weight: npred,
                        type_data: np.full([1], uri)
                    }
                    if epoch % 2 == 0:
                        __, loss_val = session.run(
                            [optimizer, loss], feed_dict=feed_dict)
                        average_loss += loss_val
                    else:
                        __, loss_val = session.run(
                            [sim_optimizer, sim_loss], feed_dict=feed_dict)
                        average_loss += loss_val

                    if step % verbose == 0:
                        average_loss /= verbose
                        print('Epoch: ', epoch, ' Average loss at step ',
                              step, ': ', average_loss)
                        writer.write('Epoch: ' + str(epoch) + ' Average loss at step ' +
                                     str(step) + ': ' + str(average_loss) + '\n')
                        average_loss = 0
                if len(data[i]) > 0:
                    average_loss /= ((len(data[i]) %
                                      (verbose * batchSize)) / batchSize)
                    print('Epoch: ', epoch, ' Average loss at step ',
                          step, ': ', average_loss)
                    writer.write('Epoch: ' + str(epoch) + ' Average loss at step ' +
                                 str(step) + ': ' + str(average_loss) + '\n')

            end_time_epoch = dt.datetime.now()
            print("Training time took {} seconds to run 1 epoch".format(
                (end_time_epoch - start_time_epoch).total_seconds()))
            writer.write("Training time took {} seconds to run 1 epoch\n".format(
                (end_time_epoch - start_time_epoch).total_seconds()))

            if (epoch + 1) % 10 == 0:
                start_time_epoch = dt.datetime.now()
                sim = similarity.eval()
                MRR, mean_rank, hits_at_10, hits_at_1 = metric(
                    test_answer, sim, entity_kb1_vocab, top_k)

                print "MRR: ", MRR
                writer.write("MRR: " + str(MRR) + "\n")
                print "Mean Rank: ", mean_rank, " of ", len(entity_kb1_vocab)
                writer.write("Mean Rank: " + str(mean_rank) +
                             " of " + str(len(entity_kb1_vocab)) + "\n")
                print "Hits @ " + str(top_k) + ": ", hits_at_10
                writer.write("Hits @ " + str(top_k) +
                             ": " + str(hits_at_10) + "\n")
                print "Hits @ " + str(1) + ": ", hits_at_1
                writer.write("Hits @ " + str(1) + ": " + str(hits_at_1) + "\n")
                end_time_epoch = dt.datetime.now()
                print("Testing time took {} seconds.".format(
                    (end_time_epoch - start_time_epoch).total_seconds()))
                writer.write("Testing time took {} seconds.\n\n".format(
                    (end_time_epoch - start_time_epoch).total_seconds()))
                print


# In[ ]:


start_time = dt.datetime.now()
writer = open('log_imdb_tmdb.txt', 'w', 0)
run(writer, tfgraph, totalEpoch)
end_time = dt.datetime.now()
print("Training time took {} seconds to run {} epoch".format(
    (end_time - start_time).total_seconds(), totalEpoch))
writer.write("Total Training time took {} seconds to run {} epoch".format(
    (end_time - start_time).total_seconds(), totalEpoch))
writer.close()