#!/usr/bin/env python
# coding: utf-8

# In[1]:


# Copyright 2019-present NAVER Corp.
# Apache License v2.0

# Wonseok Hwang
# Sep30, 2018
import os, sys, argparse, re, json,ujson
import pandas as pd
import numpy as np
from matplotlib.pylab import *
import tensorflow as tf
import random as python_random
# import torchvision.datasets as dsets
import tensorflow_hub as hub

from sqlova.utils.utils_wikisql import *
from sqlova.model.nl2sql.wikisql_models import *
from sqlnet.dbengine import DBEngine
import bert.tokenization as tokenization
from bert.modeling import BertConfig, BertModel
from tensorflow.keras import backend as K
import random


# In[2]:


# In[14]:


# Build model
def weighted_categorical_crossentropy(weights):
    """
    A weighted version of keras.objectives.categorical_crossentropy
    
    Variables:
        weights: numpy array of shape (C,) where C is the number of classes
    
    Usage:
        weights = np.array([0.5,2,10]) # Class one at 0.5, class 2 twice the normal weights, class 3 10x.
        loss = weighted_categorical_crossentropy(weights)
        model.compile(loss=loss,optimizer='adam')
    """
    
    #weights = K.variable(weights)
        
    def loss(y_true, y_pred):
        # scale predictions so that the class probas of each sample sum to 1
        y_pred /= K.sum(y_pred, axis=-1, keepdims=True)
        # clip to prevent NaN's and Inf's
        y_pred = K.clip(y_pred, K.epsilon(), 1 - K.epsilon())
        # calc
        loss = y_true * K.log(y_pred) * weights
        loss = -K.sum(loss, -1)
        return loss
    
    return loss

def build_model_baseline(max_seq_length,max_out_len): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    weight_input = tf.keras.layers.Input(shape=(max_out_len,), name="ce_weights")
    
    bert_inputs = [in_id, in_mask, in_segment]
    all_inputs = [in_id, in_mask, in_segment, weight_input]
    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)

    denseSC = tf.keras.layers.Dense(256, activation='relu',name="denseSC")(bert_output)
    denseSA = tf.keras.layers.Dense(256, activation='relu',name="denseSA")(bert_output)
    denseWN = tf.keras.layers.Dense(256, activation='relu',name="denseWN")(bert_output)
    denseWC = tf.keras.layers.Dense(256, activation='relu',name="denseWC")(bert_output)
    denseWO = tf.keras.layers.Dense(256, activation='relu',name="denseWO")(bert_output)
    
    
    sc = tf.keras.layers.Dense(13, activation='softmax',name="sc_output")(denseSC)
    sa = tf.keras.layers.Dense(4, activation='softmax',name="sa_output")(denseSA)
    wn = tf.keras.layers.Dense(4, activation='softmax',name="wn_output")(denseWN)    
    wc = tf.keras.layers.Dense(max_out_len, activation='sigmoid',name="wc_output")(denseWC)
    wo = tf.keras.layers.Dense(max_out_len, activation='sigmoid',name="wo_output")(denseWO)

    loss = weighted_categorical_crossentropy(weight_input)    
    #loss=,
    model = tf.keras.models.Model(inputs=all_inputs, outputs=[sc,sa,wn,wc,wo])
    

    model.compile(loss={"sc_output":"sparse_categorical_crossentropy",
                        "sa_output":"sparse_categorical_crossentropy",
                        "wn_output":"sparse_categorical_crossentropy",
                        'wc_output': loss, 
                        'wo_output': loss}, optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model
    


def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


class BertLayer(tf.layers.Layer):
    def __init__(self, n_fine_tune_layers=10, **kwargs):
        self.n_fine_tune_layers = n_fine_tune_layers
        self.trainable = True
        self.output_size = 768
        super(BertLayer, self).__init__(**kwargs)

    def build(self, input_shape):
        self.bert = hub.Module(
            bert_path,
            trainable=self.trainable,
            name="{}_module".format(self.name)
        )
        trainable_vars = self.bert.variables
        
        # Remove unused layers
        trainable_vars = [var for var in trainable_vars if not "/cls/" in var.name]
        
        # Select how many layers to fine tune
        trainable_vars = trainable_vars[-self.n_fine_tune_layers :]
        
        # Add to trainable weights
        for var in trainable_vars:
            self._trainable_weights.append(var)
        
        # Add non-trainable weights
        for var in self.bert.variables:
            if var not in self._trainable_weights:
                self._non_trainable_weights.append(var)
        
        super(BertLayer, self).build(input_shape)

    def call(self, inputs):
        inputs = [K.cast(x, dtype="int32") for x in inputs]
        input_ids, input_mask, segment_ids = inputs
        bert_inputs = dict(
            input_ids=input_ids, input_mask=input_mask, segment_ids=segment_ids
        )
        result = self.bert(inputs=bert_inputs, signature="tokens", as_dict=True)
        pooled_output = result['pooled_output']
        seq_output = result['sequence_output']
        
        return pooled_output, seq_output

    def compute_output_shape(self, input_shape):
        return (input_shape[0], self.output_size)


# In[6]:


bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=True)


# In[9]:


def create_tokenizer_from_hub_module(bert_hub_module_handle):
    """Get the vocab file and casing info from the Hub module."""
    with tf.Graph().as_default():
        tf.print('GRAPH STARTED')
        bert_module = hub.Module(bert_hub_module_handle)
        tf.print('MODULE DOWNLOADED')
        tokenization_info = bert_module(signature="tokenization_info", as_dict=True)
        tf.print('TOKENIZATION INFO LOADED')
        with tf.Session() as sess:
            tf.print('SESSION STARTED')
            vocab_file, do_lower_case = sess.run([tokenization_info["vocab_file"],
                                                tokenization_info["do_lower_case"]])
            tf.print('VARIABLES RAN')
            return FullTokenizer(
                vocab_file=vocab_file, do_lower_case=do_lower_case)


# ## BERT FORMATTING AND OUTPUT GENERATION

# In[10]:


def generate_inputs(tokenizer, nlu1_tok, hds1):
   tokens = []
   segment_ids = []
   tokens.append("[CLS]")

   segment_ids.append(0)
   for token in nlu1_tok:
       tokens.append(token)
       segment_ids.append(0)
   tokens.append("[SEP]")
   segment_ids.append(0)

   header_break = [len(tokens),len(tokens) + len(hds1)]
   # for doc
   for i, hds11 in enumerate(hds1):
       sub_tok = tokenizer.tokenize(hds11)
       tokens += sub_tok
       segment_ids += [1] * len(sub_tok)
       if i < len(hds1)-1:
           tokens.append("[SEP]")
           segment_ids.append(0)
       elif i == len(hds1)-1:
           tokens.append("[SEP]")
           segment_ids.append(1)
       else:
           raise EnvironmentError


   return tokens, segment_ids, header_break

def _formatForBert(tokenizer, query_token, header_token, max_seq_length):
    #####For each example, tokenize with BERT tokenizer
    #Mark each example begining, and separate each header 
    #so BERT understands the query is a meaningful sequence of words but the headers are not
    #concatenate the tokens with breaks but preserve where the headers stop and start
    #then pad each example to the max sequence length within each batch
    double_tokenized_tokens = []
    for (i, token) in enumerate(query_token):
            #t_to_tt_idx1.append(
            #    len(double_tokenized_tokens))  # all_doc_tokens[ indicate the start position of original 'white-space' tokens.
            sub_tokens = tokenizer.tokenize(token)
            for sub_token in sub_tokens:
            #    tt_to_t_idx1.append(i)
                double_tokenized_tokens.append(sub_token)  # all_doc_tokens are further tokenized using WordPiece tokenizer

    tokens1, segment_ids1, header_break = generate_inputs(tokenizer, double_tokenized_tokens, header_token)
    input_ids1 = tokenizer.convert_tokens_to_ids(tokens1)

    # Input masks
    # The mask has 1 for real tokens and 0 for padding tokens. Only real
    # tokens are attended to.
    input_mask1 = [1] * len(input_ids1)

        # 3. Zero-pad up to the sequence length.
    while len(input_ids1) < max_seq_length:
            input_ids1.append(0)
            input_mask1.append(0)
            segment_ids1.append(0)

    assert len(input_ids1) == max_seq_length
    assert len(input_mask1) == max_seq_length
    assert len(segment_ids1) == max_seq_length
    return input_ids1,tokens1,segment_ids1,input_mask1,header_break

def formatForBert(train_data,BERT_PT_PATH= "/DataDrive/master-wikisql/annotated_data/",bert_type="uncased_L-12_H-768_A-12",max_seq_length=222):
    
    input_ids = []
    tokens = []
    segment_ids = []
    input_masks = []
    header_breaks = []

    bert_config_file = os.path.join(BERT_PT_PATH, f'bert_config_{bert_type}.json')
    vocab_file = os.path.join(BERT_PT_PATH, f'vocab_{bert_type}.txt')

    bert_config = BertConfig.from_json_file(bert_config_file)
    tokenizer = tokenization.FullTokenizer(
        vocab_file=vocab_file, do_lower_case=True)
    
    for train_example in train_data:
        query_token = train_example["query"]
        header_token = train_example["header"]
        input_ids1, tokens1, segment_ids1, input_mask1, header_break =                 _formatForBert(tokenizer, query_token,header_token,max_seq_length)
        input_ids.append(input_ids1)
        tokens.append(tokens1)
        segment_ids.append(segment_ids1)
        input_masks.append(input_mask1)
        header_breaks.append(header_break)

    #TODO return header breaks and use it too
    #bert_inputs = []
    return np.array(input_ids), np.array(input_masks), np.array(segment_ids)

def getBertOutput(bert_inputs, max_seq_length):
    all_input_ids = tf.keras.layers.Input(shape=(max_seq_length,), name = "input_ids")
    all_input_mask = tf.keras.layers.Input(shape=(max_seq_length,), name = "input_masks")
    all_segment_ids = tf.keras.layers.Input(shape=(max_seq_length,), name = "segment_ids")
    bert_output = BertLayer(n_fine_tune_layers=10)(bert_inputs)


# ## READING INPUT, PARSING INTO WHAT WILL BE FED TO THE MODEL

# In[11]:


def makeTrainingData(wikisql_path = "/DataDrive/master-wikisql/annotated_data/",sample_size=32,use_reduced_set=False):
    train_data, train_table, dev_data, dev_table, _, _ = load_wikisql(wikisql_path,
                                                                  use_reduced_set, sample_size, no_w2i=True, no_hs_tok=True)
    
    return parseToTextLines(train_data,train_table)

def parseToTextLines(train_data,train_table):
    train_dataframe = pd.DataFrame(train_data)
    question_toks = [[x.lower() for x in sublist] for sublist in train_dataframe["question_tok"].tolist()]
    sql = train_dataframe["sql"].tolist()
    table_ids = train_dataframe["table_id"].tolist()
    table_headers = [[x.lower() for x in train_table[tid]["header"]] for tid in table_ids]
    labels = parseSqlToLabels(sql)
    assert len(question_toks)==len(table_headers)
    
    keras_model_train_data = [{"query":question_toks[i],"header":table_headers[i]} for i in range(len(question_toks))]
    return keras_model_train_data, labels


# In[12]:


def load_wikisql(path_wikisql, toy_model, toy_size, bert=False, no_w2i=False, no_hs_tok=False, aug=False):
    # Get data
    train_data, train_table = load_wikisql_data(path_wikisql, mode='train', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok, aug=aug)
    dev_data, dev_table = load_wikisql_data(path_wikisql, mode='dev', toy_model=toy_model, toy_size=toy_size, no_hs_tok=no_hs_tok)


    # Get word vector
    if no_w2i:
        w2i, wemb = None, None
    else:
        w2i, wemb = load_w2i_wemb(path_wikisql, bert)


    return train_data, train_table, dev_data, dev_table, w2i, wemb


def load_wikisql_data(path_wikisql, mode='train', toy_model=False, toy_size=10, no_hs_tok=False, aug=False):
    """ Load training sets
    """
    if aug:
        mode = f"aug.{mode}"
        print('Augmented data is loaded!')

    path_sql = os.path.join(path_wikisql, mode+'_tok.jsonl')
    if no_hs_tok:
        path_table = os.path.join(path_wikisql, mode + '.tables.jsonl')
    else:
        path_table = os.path.join(path_wikisql, mode+'_tok.tables.jsonl')

    data = []
    table = {}
    with open(path_sql) as f:
        for idx, line in enumerate(f):
            if toy_model and idx >= toy_size:
                break

            t1 = json.loads(line.strip())
            data.append(t1)

    with open(path_table) as f:
        for idx, line in enumerate(f):
            if toy_model and idx > toy_size:
                break

            t1 = json.loads(line.strip())
            table[t1['id']] = t1

    return data, table


# ## FORMATTING SQL LABELS

# In[13]:


def get_g(sql_i):
    """ for backward compatibility, separated with get_g"""
    g_sc = []
    g_sa = []
    g_wn = []
    g_wc = []
    g_wo = []
    g_wv = []
    for b, psql_i1 in enumerate(sql_i):
        g_sc.append( psql_i1["sel"] )
        g_sa.append( psql_i1["agg"])

        conds = psql_i1['conds']
        if not psql_i1["agg"] < 0:
            g_wn.append( len( conds ) )
            g_wc.append( get_wc1(conds) )
            g_wo.append( get_wo1(conds) )
            g_wv.append( get_wv1(conds) )
        else:
            raise EnvironmentError
    return g_sc, g_sa, g_wn, g_wc, g_wo, g_wv

def padWhereConditions(clause_list):
    b = np.zeros([len(clause_list),len(max(clause_list,key = lambda x: len(x)))])
    b[:] = -1
    for i,j in enumerate(clause_list):
        b[i][0:len(j)] = j
    return(b)

def parseSqlToLabels(sql): 
    #return sql
    #return [random.randint(0, 1) for x in sql]
    sc,sa,wn,wc,wo,wv = get_g(sql)
    wc,wo = padWhereConditions(wc),padWhereConditions(wo)
    return np.array(sc),np.array(sa),np.array(wn),wc,wo
    
#############new or modified


#define token start and stop  positions
def tok_hdr_start_top(input_ids):
	tok_pos = []
	head_pos = []
	for idx, row in enumerate(input_ids):
		#print(idx)
		#seg positions
		segs_long = np.where(row == 102)[0]
		tok_pos.append([1, segs_long[0]]) #start position is 1, ignore [CLS] token
		segs = [];
		for s in range(1,len(segs_long)):
			start = segs_long[s-1] + 1 # start position
			end = segs_long[s] # end position
			segs.append([start,end])
			
		head_pos.append(segs)
	tok_pos = np.array(tok_pos)
	head_pos = np.array(head_pos)
	return tok_pos, head_pos

def create_wemb_masks(tok_pos, head_pos, mask_shape):
	tok_mask = np.zeros(mask_shape)
	head_mask = np.zeros(mask_shape)
	
	for tidx, tp in enumerate(tok_pos):
		tok_mask[tidx, tp[0]:tp[1],:] = 1
	
	for hidx, hp in enumerate(head_pos):
		for hp1 in hp:
			head_mask[hidx,hp1[0]:hp1[1],:] = 1
	
	return tok_mask, head_mask
	
	
	
def build_model_masked(max_seq_length,max_out_len, wemb_mask_shape): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    weight_input = tf.keras.layers.Input(shape=(max_out_len,), name="ce_weights")
    
    wemb_n_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_n_mask")
    wemb_h_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_h_mask")
    
    
    bert_inputs = [in_id, in_mask, in_segment]
    all_inputs = [in_id, in_mask, in_segment, weight_input, wemb_n_mask, wemb_h_mask]
    pooled_output, seq_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)
    
    
    
    wemb_n = tf.keras.layers.Multiply()([seq_output, wemb_n_mask])
    wemb_h = tf.keras.layers.Multiply()([seq_output, wemb_h_mask])
    
    wemb_n = tf.keras.layers.Flatten()(wemb_n)
    wemb_h = tf.keras.layers.Flatten()(wemb_h)
    ####mask and shit here
    
    denseSC = tf.keras.layers.Dense(256, activation='relu',name="denseSC")(wemb_h)
    denseSA = tf.keras.layers.Dense(256, activation='relu',name="denseSA")(wemb_h)
    denseWN = tf.keras.layers.Dense(256, activation='relu',name="denseWN")(pooled_output)
    denseWC = tf.keras.layers.Dense(256, activation='relu',name="denseWC")(wemb_h)
    denseWO = tf.keras.layers.Dense(256, activation='relu',name="denseWO")(wemb_h)
    
    
    sc = tf.keras.layers.Dense(13, activation='softmax',name="sc_output")(denseSC)
    sa = tf.keras.layers.Dense(4, activation='softmax',name="sa_output")(denseSA)
    wn = tf.keras.layers.Dense(4, activation='softmax',name="wn_output")(denseWN)    
    wc = tf.keras.layers.Dense(max_out_len, activation='sigmoid',name="wc_output")(denseWC)
    wo = tf.keras.layers.Dense(max_out_len, activation='sigmoid',name="wo_output")(denseWO)

    loss = weighted_categorical_crossentropy(weight_input)    
    #loss=,
    model = tf.keras.models.Model(inputs=all_inputs, outputs=[sc,sa,wn,wc,wo])
    

    model.compile(loss={"sc_output":"sparse_categorical_crossentropy",
                        "sa_output":"sparse_categorical_crossentropy",
                        "wn_output":"sparse_categorical_crossentropy",
                        'wc_output': loss, 
                        'wo_output': loss}, optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model
    


if __name__ == "__main__":

	num_CPU = 1
	num_GPU = 0
	num_cores = 2
	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
						inter_op_parallelism_threads=num_cores, 
						allow_soft_placement=True,
						device_count = {'CPU' : num_CPU,
										'GPU' : num_GPU}
					   )
                       
	sess = tf.Session(config = config)
	
	os.environ['TFHUB_CACHE_DIR'] = '/DataDrive/master-wikisql/tfhub_cache/'


	# In[3]:


	bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"


	# In[4]:


	#qsess = tf.Session()
	training_data, train_labels = makeTrainingData(use_reduced_set=True,sample_size=2000)
	sc,sa,wn,wc,wo = train_labels
	w = np.ones_like(wc.copy())
	#w[np.where(w>=0)] = 1
	w[np.where(wc==-1)] = 0
	max_out_len = wc.shape[1]
	split = int(0.8*len(wc))
	test_data = training_data[split:]
	training_data = training_data[:split]

	test_labels_sc = sc[split:]
	train_labels_sc = sc[:split]

	test_labels_sa = sa[split:]
	train_labels_sa = sa[:split]


	test_labels_wn = wn[split:]
	train_labels_wn = wn[:split]

	test_labels_wc = wc[split:]
	train_labels_wc = wc[:split]

	test_labels_wo = wo[split:]
	train_labels_wo = wo[:split]

	train_weights = w[:split]
	test_weights = w[split:]
	#test_labels = np.array(test_labels).reshape(-1, 1)
	#train_labels= np.array(train_labels).reshape(-1, 1)

	train_input_ids,train_input_masks,train_segment_ids = formatForBert(training_data)
	test_input_ids,test_input_masks,test_segment_ids = formatForBert(test_data)
	# In[19]:
	#training_data,train_labels = makeTrainingData(use_reduced_set=True,sample_size=2000)
	# In[16]:
	# Build the rest of the classifier 
	
	
	
	max_seq_length = 222
	bert_out_size = 768
	
	#header start stopcreate masks for train and test
	train_mask_shape = (len(train_input_ids), max_seq_length, bert_out_size)
	train_tok_pos, train_head_pos = tok_hdr_start_top(train_input_ids)
	train_tok_mask, train_head_mask = create_wemb_masks(train_tok_pos, train_head_pos, train_mask_shape)
	
	
	test_mask_shape = (len(test_input_ids), max_seq_length, bert_out_size)
	test_tok_pos, test_head_pos = tok_hdr_start_top(test_input_ids)
	test_tok_mask, test_head_mask = create_wemb_masks(test_tok_pos, test_head_pos, test_mask_shape)
	
	
	model = build_model_masked(max_seq_length=max_seq_length, max_out_len=max_out_len, wemb_mask_shape = tuple(test_mask_shape[1:]))

	# Instantiate variables
	initialize_vars(sess)

	model.fit(
		[train_input_ids, train_input_masks, train_segment_ids, train_weights, train_tok_mask, train_head_mask], 
			{"sc_output":train_labels_sc,"sa_output":train_labels_sa,
			 "wn_output":train_labels_wn,"wc_output":train_labels_wc,"wo_output":train_labels_wo},
		validation_data=([test_input_ids,test_input_masks,test_segment_ids, test_weights, test_tok_mask, test_head_mask],
						 {"sc_output":test_labels_sc,"sa_output":test_labels_sa,
						  "wn_output":test_labels_wn,"wc_output":test_labels_wc,
						 "wo_output":test_labels_wo}),
		epochs=1,
		batch_size=32
	)


	# In[18]:


	i = 1
	test_example = test_data[i]
	input1,input2,input3,input4 = test_input_ids,test_input_masks,test_segment_ids[i], test_weights[i]


	# In[19]:


	model.predict([input1,input2,input3,input4])


	# In[89]:


	{"sc_output":train_labels_sc,"sa_output":train_labels_sa,
	 "wn_output":train_labels_wn,"wc_output":train_labels_wc,"wo_output":train_labels_wo},
	train_labels_wo.max()



