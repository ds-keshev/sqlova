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


# In[3]:



# In[20]:


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

def build_model(max_seq_length,max_out_len,max_header_length=13): 
    in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
    in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
    in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
    #weight_input = tf.keras.layers.Input(shape=(max_out_len,), name="ce_weights")
    
    bert_inputs = [in_id, in_mask, in_segment]
    all_inputs = [in_id, in_mask, in_segment]
    bert_output = BertLayer(n_fine_tune_layers=3)(bert_inputs)

    denseSC = tf.keras.layers.Dense(256, activation='relu',name="denseSC")(bert_output)
    denseSA = tf.keras.layers.Dense(256, activation='relu',name="denseSA")(bert_output)
    denseWN = tf.keras.layers.Dense(256, activation='relu',name="denseWN")(bert_output)
    denseWC = tf.keras.layers.Dense(256, activation='relu',name="denseWC")(bert_output)
    denseWO = tf.keras.layers.Dense(256, activation='relu',name="denseWO")(bert_output)
    
    
    sc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="sc_output")(denseSC)
    sa = tf.keras.layers.Dense(4, activation='softmax',name="sa_output")(denseSA)
    wn = tf.keras.layers.Dense(4, activation='softmax',name="wn_output")(denseWN)    
    wc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="wc_output")(denseWC)
    wo = tf.keras.layers.Dense(3, activation='softmax',name="wo_output")(denseWO)

    #loss = weighted_categorical_crossentropy(weight_input)    
    #loss=,
    model = tf.keras.models.Model(inputs=all_inputs, outputs=[sc,sa,wn,wc,wo])
    

    model.compile(loss={"sc_output":"sparse_categorical_crossentropy",
                        "sa_output":"sparse_categorical_crossentropy",
                        "wn_output":"sparse_categorical_crossentropy",
                        #'wc_output': loss, 
                        #'wo_output': loss}, optimizer='adam', metrics=['accuracy'])
                        'wc_output': 'categorical_crossentropy', 
                        'wo_output': 'categorical_crossentropy'}, optimizer='adam', metrics=['accuracy'])
    model.summary()
    
    return model

def initialize_vars(sess):
    sess.run(tf.local_variables_initializer())
    sess.run(tf.global_variables_initializer())
    sess.run(tf.tables_initializer())
    K.set_session(sess)


# In[128]:


#test_labels_wc.shape


# ## DEFINE THE ACTUAL BERT LAYER
# 
# 
# 
# 

# In[12]:




# In[6]:


bert_module = hub.Module("https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1", trainable=True)


# In[11]:


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

# In[9]:


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


# In[8]:


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

# In[7]:


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
def multihotEncodeWhereThings(where_things):
    max_where_things = len(set([x for y in where_things for x in y]))
    where_things_multihot = np.zeros([len(where_things),max_where_things])
    for i, where_things_locs in enumerate(where_things):
        where_things_multihot[i][where_things_locs] = 1
        
    return where_things_multihot
    
def parseSqlToLabels(sql): 
    #return sql
    #return [random.randint(0, 1) for x in sql]
    sc,sa,wn,wc,wo,wv = get_g(sql)
    wc,wo = multihotEncodeWhereThings(wc),multihotEncodeWhereThings(wo)
    return np.array(sc),np.array(sa),np.array(wn),wc,wo


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
			#segs.append([start,end])
			segs.append([start, start+1])
			
		head_pos.append(segs)
	tok_pos = np.array(tok_pos)
	head_pos = np.array(head_pos)
	return tok_pos, head_pos

def create_wemb_masks(tok_pos, head_pos, mask_shape):
	tok_mask = np.zeros(mask_shape)
	head_mask = np.zeros(mask_shape)
	head_col_mask = np.ones(mask_shape)
	for tidx, tp in enumerate(tok_pos):
		tok_mask[tidx, tp[0]:tp[1],:] = 1
	
	for hidx, hp in enumerate(head_pos):
		for hp1 in hp:
			head_mask[hidx,hp1[0]:hp1[1],:] = 1
	
	for hidx, hp in enumerate(head_pos):
		for hp1 in hp:
			head_col_mask[hidx,hp1[0]:hp1[1],:] = 1
	
	return tok_mask, head_mask, head_col_mask
	
	
	
def mask_func(x):
	bool_mask = tf.logical_not(tf.greater(x,1))
	mask = tf.ones_like(x)*-9999999.0
	x = tf.where(bool_mask, x, mask)
	return x


#def mask_func(x):
#	x = tf.slice(x,[0,0,0],[x.shape[0], x.shape[1],1])
#	bool_mask = tf.greater(x,0)
	
#	mask = tf.ones_like(x)*-9999999.0
#	x = tf.where(bool_mask, x, mask)
#	return x
	
def output_hack(x):
	bool_mask = tf.greater(x,0)
	non_zero_values = tf.gather_nd(x, tf.where(boolean_mask))
	rows = tf.split(non_zero_values, n_non_zero)
	#print(rows)
	# Pad with zeros wherever necessary and recombine into a single tensor
	out = tf.stack([tf.argmax(r)for r in rows])
	#rows = tf.split(non_zero_values, n_non_zero)
	
def build_model_masked(max_seq_length,max_out_len, wemb_mask_shape): 
	max_header_length = 13
	tuple(test_mask_shape[1:])
	in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
	in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
	in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
	weight_input = tf.keras.layers.Input(shape=(max_out_len,), name="ce_weights")

	wemb_n_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_n_mask")
	wemb_h_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_h_mask")
	wemb_h_col_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_h_col_mask")

	bert_inputs = [in_id, in_mask, in_segment]
	all_inputs = [in_id, in_mask, in_segment, wemb_n_mask, wemb_h_mask, wemb_h_col_mask]
	pooled_output, seq_output = BertLayer(n_fine_tune_layers=0, trainable = False)(bert_inputs)
	print(seq_output)

	#wemb_n = tf.keras.layers.Multiply()([seq_output, wemb_n_mask])
	#wemb_h = tf.keras.layers.Multiply()([seq_output, wemb_h_mask])
	#wemb_h_col = tf.keras.layers.Multiply()([seq_output, wemb_h])
	#wemb_h_col = tf.keras.layers.Lambda(lambda x: mask_func(x))(wemb_h)

	#wemb_h_col = slice_concat()(wemb_h)
	#print(wemb_h_col)
	#wemb_n = tf.keras.layers.Flatten()(wemb_n)
	#wemb_h = tf.keras.layers.Flatten()(wemb_h)
	wemb_h_col = tf.keras.layers.Flatten()(seq_output)
	print(wemb_h_col)
	####mask and shit here

	#denseSC = tf.keras.layers.Dense(256, activation='relu',name="denseSC")(wemb_h_col)
	#denseSA = tf.keras.layers.Dense(4, activation='relu',name="denseSA")(wemb_h)
	denseSA = tf.keras.layers.Dense(4, activation='relu',name="denseSA")(pooled_output)

	denseWN = tf.keras.layers.Dense(4, activation='relu',name="denseWN")(pooled_output)
	#denseWC = tf.keras.layers.Dense(256, activation='relu',name="denseWC")(wemb_h_col)
	#denseWO = tf.keras.layers.Dense(4, activation='relu',name="denseWO")(wemb_h)
	denseWO = tf.keras.layers.Dense(4, activation='relu',name="denseWO")(pooled_output)


	sc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="sc_output")(wemb_h_col)
	sa = tf.keras.layers.Dense(4, activation='softmax',name="sa_output")(denseSA)
	wn = tf.keras.layers.Dense(4, activation='softmax',name="wn_output")(denseWN)    
	wc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="wc_output")(denseWN)
	wo = tf.keras.layers.Dense(3, activation='softmax',name="wo_output")(denseWO)

	#loss = weighted_categorical_crossentropy(weight_input)    
	#loss=,
	model = tf.keras.models.Model(inputs=all_inputs, outputs=[sc,sa,wn,wc,wo])


	model.compile(loss={"sc_output":"sparse_categorical_crossentropy",
						"sa_output":"sparse_categorical_crossentropy",
						"wn_output":"sparse_categorical_crossentropy",
						#'wc_output': loss, 
						#'wo_output': loss}, 
						'wc_output': 'categorical_crossentropy', 
						'wo_output': 'categorical_crossentropy'
						}, optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model


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
		print('CONCAAAT')
		#seq_output = tf.map_fn(lambda x:remap(x), seq_output,infer_shape=False)
		#seq_output = seq_output[:,:,0]
		#seq_output = tf.reshape(seq_output, shape = (32, 222))
		#slice_concat()(wemb_h)
		return pooled_output, seq_output

	def compute_output_shape(self, input_shape):
		return (input_shape[0], self.output_size)


def build_model_masked_select(max_seq_length,max_out_len, wemb_mask_shape, mask_idx): 
	max_header_length = 13
	#tuple(test_mask_shape[1:])
	in_id = tf.keras.layers.Input(shape=(max_seq_length,), name="input_ids")
	in_mask = tf.keras.layers.Input(shape=(max_seq_length,), name="input_masks")
	in_segment = tf.keras.layers.Input(shape=(max_seq_length,), name="segment_ids")
	weight_input = tf.keras.layers.Input(shape=(max_out_len,), name="ce_weights")

	wemb_n_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_n_mask")
	wemb_h_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_h_mask")
	wemb_h_col_mask = tf.keras.layers.Input(shape=wemb_mask_shape, name="wemb_h_col_mask")
	wemb_mask_idx= tf.keras.layers.Input(shape=tuple(wemb_mask_shape[1:]), name="wemb_h_col_mask")
	
	output_mask = tf.keras.layers.Input(shape = (max_header_length,), name = 'output_mask')
	
	
	bert_inputs = [in_id, in_mask, in_segment]
	all_inputs = [in_id, in_mask, in_segment, wemb_n_mask, wemb_h_mask, wemb_h_col_mask, output_mask]
	pooled_output, seq_output = BertLayer(n_fine_tune_layers=1, trainable = False)(bert_inputs)
	
	#test = slice_concat()
	#test.build(mask_idx)
	#test = test(seq_output)
	#print(test)
	#wemb = tf.keras.layers.Lambda(lambda x: K.map_fn(lambda y:remap(y),x),output_shape=[32,222,768])(seq_output)
	
	
	#wemb = tf.reshape(wemb, shape = [tf.shape(seq_output)[0], 222, 768])
	#print(wemb)
	#print('BEFORE')
	#print(wemb)
	#wemb = tf.keras.layers.Reshape((-1, 222,768))(wemb)
	#print(wemb)
	#print(seq_output)

	#wemb_n = tf.keras.layers.Multiply()([seq_output, wemb_n_mask])
	#wemb_h = tf.keras.layers.Multiply()([seq_output, wemb_h_mask])
	#wemb_h = 
	
	test = slice_concat()
	test.build(mask_idx)
	test = test(seq_output)
	#wemb_h = tf.slice_concat()(wemb_h)
	#wemb_h_col = tf.keras.layers.Multiply()([seq_output, wemb_h])
	#wemb_h_col = tf.keras.layers.Lambda(lambda x: mask_func(x))(wemb_h)

	#wemb_h_col = slice_concat()(wemb_h)
	#print(wemb_h_col)
	#wemb_n = tf.keras.layers.Flatten()(wemb_n)
	#wemb_h = tf.keras.layers.Flatten()(wemb_h)
	#wemb_h_col = tf.keras.layers.Flatten()(seq_output)
	#print(wemb_h_col)
	####mask and shit here

	denseSC = tf.keras.layers.Dense(1024, activation='relu',name="denseSC")(test)
	#denseSC1 = tf.keras.layers.Dense(1024, activation='relu',name="denseSC1")(denseSC)
	#denseSC2 = tf.keras.layers.Dense(1024, activation='relu',name="denseSC2")(denseSC1)
	
	
	#denseSA = tf.keras.layers.Dense(4, activation='relu',name="denseSA")(wemb_h)
	#denseSA = tf.keras.layers.Dense(4, activation='relu',name="denseSA")(pooled_output)

	#denseWN = tf.keras.layers.Dense(4, activation='relu',name="denseWN")(pooled_output)
	#denseWC = tf.keras.layers.Dense(256, activation='relu',name="denseWC")(wemb_h_col)
	#denseWO = tf.keras.layers.Dense(4, activation='relu',name="denseWO")(wemb_h)
	#denseWO = tf.keras.layers.Dense(4, activation='relu',name="denseWO")(pooled_output)
	
	#pre_sc = tf.keras.layers.Dense(max_header_length, activation='linear',name="sc_pre_output")(denseSC2)
	#pre_sc = tf.keras.layers.Add()([pre_sc, output_mask])
	
	#sc = tf.keras.layers.Activation(activation = 'softmax', name  = 'sc_output')(pre_sc)
	sc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="sc_output")(denseSC)
	#sa = tf.keras.layers.Dense(4, activation='softmax',name="sa_output")(denseSA)
	#wn = tf.keras.layers.Dense(4, activation='softmax',name="wn_output")(denseWN)    
	#wc = tf.keras.layers.Dense(max_header_length, activation='softmax',name="wc_output")(denseWN)
	#wo = tf.keras.layers.Dense(3, activation='softmax',name="wo_output")(denseWO)

	#loss = weighted_categorical_crossentropy(weight_input)    
	#loss=,
	model = tf.keras.models.Model(inputs=all_inputs, outputs=[sc])


	model.compile(loss={"sc_output":"sparse_categorical_crossentropy",
						#"sa_output":"sparse_categorical_crossentropy",
						#"wn_output":"sparse_categorical_crossentropy",
						#'wc_output': loss, 
						#'wo_output': loss}, 
						#'wc_output': 'categorical_crossentropy', 
						#'wo_output': 'categorical_crossentropy'
						}, optimizer='adam', metrics=['accuracy'])
	model.summary()

	return model, seq_output
    

###############################
###############################
def pad_values(a_as_vector, max_n = 20):
   print(a_as_vector)
   zero_padding = tf.zeros(max_n - tf.shape(a_as_vector), dtype=a_as_vector.dtype)
   # Concatenate `a_as_vector` with the padding.
   a_padded = K.concatenate([a_as_vector, zero_padding], 0)
   print(a_padded)
   return a_padded
  
def remap(vector_a):
  b = K.map_fn(lambda x:pad_values(tf.gather(x,tf.squeeze(tf.where(tf.not_equal(x,0))))),vector_a)
  return b
  
def remap_old(vector_a):
   b = tf.map_fn(lambda x:pad_values(tf.gather(x,tf.squeeze(tf.where(tf.not_equal(x,0))))),vector_a)
  # b = b.slice(start = [0,0], 
   return b

class slice_concat(tf.layers.Layer):
	def __init__(self):
		super(slice_concat, self).__init__()
	def build(self, mask_idx):
		self.mask_idx = mask_idx
		#self.max_n = input_params[1]
		#print(input_params)
		super(slice_concat, self).build(mask_idx)
	def call(self, inputs):
		out_arr = tf.gather_nd(inputs, self.mask_idx)
		print(out_arr)
		#out_arr = tf.stack(tf.map_fn(lambda x:remap(x), inputs,infer_shape=False))
		#print(out_arr)
		#out_arr = tf.map_fn(lambda x: slice_map(x,self.input_idx), inputs)
		#slice_x = tf.slice(inputs, begin = [0,0,0], size = [tf.shape(inputs)[-1], 222, 0])
		#out_arr = tf.map_fn(lambda x:pad_values(tf.gather(x,tf.squeeze(tf.where(tf.not_equal(x,0))))),inputs)
		#out_arr = tf.stack(out_arr)
		#tf.print(tf.shape(out_arr))
		#out_arr = tf.reshape(out_arr, shape = (tf.shape(inputs)[0],222,768))
		#tf.print(tf.shape(out_arr))
		##out_arr = out_arr[:,:,0]
		#tf.print(tf.shape(out_arr))
		#out_arr = tf.squeeze(out_arr)
		#tf.print(tf.shape(out_arr))
		#out_arr = tf.reshape(out_arr, shape = (tf.shape(inputs)[0],222))
		#tf.print(tf.shape(out_arr))
		#print(out_arr)
		#out_arr = tf.slice(out_arr, begin = [0,0,0], size = tf.shape(out_arr)[-1], max_n, tf
		return out_arr
	def compute_output_shape(self, input_shape):
		return (input_shape[0], 2)
		
class custom_mask(tf.layers.Layer):
	def __init_(self):
		super(custom_mask,self).__init__()
	def build(self, mask_idx):
		self.mask_idx = mask_idx
		super(custom_mask, self).build(mask_idx)
	def call():
		return self.mask_idx
##################################

def zero_pad_mask(mask, n_pos = 20):
	mask_df = pd.DataFrame(mask)
	mask_df = mask_df[mask_df[2] == 0]
	out = [];
	
	for bid in mask_df[0].unique():
		batch_df = mask_df[mask_df[0] == bid]
		batch_arr  = batch_df.iloc[:n_pos,:].values
		n_missing = n_pos - batch_arr.shape[0]
		pad = np.tile([0,0,0], [n_missing, 1])
		batch_arr = np.concatenate((batch_arr, pad))
		batch_arr = [list(row) for row in batch_arr]
		#print(batch_arr.shape)
		out.append(batch_arr)
	
	##out = np.concatenate(batch_arr, axis = 1)
	return out
		
	
	
	
		


		

if __name__ == '__main__':
	num_CPU = 1
	num_GPU = 0
	num_cores = 10
	config = tf.ConfigProto(intra_op_parallelism_threads=num_cores,
						inter_op_parallelism_threads=num_cores,
						allow_soft_placement=True,
						device_count = {'CPU' : num_CPU,
										'GPU' : num_GPU}
					   )

	sess = tf.Session(config = config)

	training_data, train_labels = makeTrainingData(use_reduced_set=True,sample_size=10000)
	sc,sa,wn,wc,wo = train_labels
	
	# In[4]:
	os.environ['TFHUB_CACHE_DIR'] = '/DataDrive/master-wikisql/tfhub_cache/'


	# In[5]:
	bert_path = "https://tfhub.dev/google/bert_uncased_L-12_H-768_A-12/1"
	set(wc.flatten())

	# In[87]:
	max_num_headers = len(set(wc.flatten()))
	w = np.ones([wc.shape[0],wc.shape[1],max_num_headers])
	w.shape
	w[np.where(wc==-1)] = 0
	len(w)


	#training_data, train_labels = makeTrainingData(use_reduced_set=True,sample_size=10000)
	#sc,sa,wn,wc,wo = train_labels
	
	max_num_headers = len(set(wc.flatten()))
	w = np.ones([wc.shape[0],wc.shape[1]])
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

	#training_data,train_labels = makeTrainingData(use_reduced_set=True,sample_size=5000)

	
	max_seq_length = 222
	bert_out_size = 768
	batch_size = 32
	for epoch in range(10):
		for b in range(0, 8000, batch_size):
			in_train_input_ids = train_input_ids[b:b+batch_size]
			in_train_input_masks = train_input_masks[b:b+batch_size]
			in_train_segment_ids = train_segment_ids[b:b+batch_size]
			in_train_labels_sc = train_labels_sc[b:b+batch_size]
			
			#header start stopcreate masks for train and test
			train_mask_shape = (len(in_train_input_ids), max_seq_length, bert_out_size)
			train_tok_pos, train_head_pos = tok_hdr_start_top(in_train_input_ids)
			train_tok_mask, train_head_mask, train_head_col_mask = create_wemb_masks(train_tok_pos, train_head_pos, train_mask_shape)
			train_head_mask_idx = zero_pad_mask(np.array(np.where(train_head_mask)).T, n_pos = 13)
			
			output_mask = np.zeros((batch_size, 13))
			for row in range(batch_size):
				end = len(train_head_pos[row])
				output_mask[row,end:] = -999999999
			#print(output_mask)
			#x = asd
			#todo input mask idx
			#x = asd
			#test_mask_shape = (len(test_input_ids), max_seq_length, bert_out_size)
			#test_tok_pos, test_head_pos = tok_hdr_start_top(test_input_ids)
			#test_tok_mask, test_head_mask, test_head_col_mask  = create_wemb_masks(test_tok_pos, test_head_pos, test_mask_shape)
			
			layer_weights = []
			#x = sad
			#model = build_model_masked(max_seq_length=max_seq_length, max_out_len=max_out_len, wemb_mask_shape = tuple(test_mask_shape[1:]))
			
			if epoch == 0 and b == 0:
				model, seq_output = build_model_masked_select(max_seq_length=max_seq_length, max_out_len=max_out_len, wemb_mask_shape = tuple(train_mask_shape[1:]), mask_idx = train_head_mask_idx)
				initialize_vars(sess)
				
			else:
				test = slice_concat()
				test.build(train_head_mask_idx)
				test = test(seq_output)
				model.layers[4] = test
			
			#if len(layer_weights) > 0:
			#	model.layers[5].set_weights(layer_weights)
			print(epoch, '-', b)
			model.fit(
				#[train_input_ids, train_input_masks, train_segment_ids, train_weights], 
				[in_train_input_ids, in_train_input_masks, in_train_segment_ids, train_tok_mask, train_head_mask, train_head_col_mask, output_mask], 
					{"sc_output":in_train_labels_sc},
				#validation_data=(#[test_input_ids,test_input_masks,test_segment_ids, test_weights],
				#			[test_input_ids,test_input_masks,test_segment_ids, test_tok_mask, test_head_mask[:,:,0], test_head_col_mask],
				#				 {"sc_output":test_labels_sc}),
				epochs=1,
				batch_size=batch_size
			)
		
		#layer_weights = model.layers[5].get_weights()
		# In[ ]:



	# Build the rest of the classifier 
		
	#max_out_len = 3
	#model = build_model(max_seq_length=222,max_out_len=max_out_len)

	# Instantiate variables
	
	model.fit(
		#[train_input_ids, train_input_masks, train_segment_ids, train_weights], 
		[train_input_ids, train_input_masks, train_segment_ids, train_tok_mask, train_head_mask, train_head_col_mask, output_mask], 
			{"sc_output":train_labels_sc,"sa_output":train_labels_sa,
			 "wn_output":train_labels_wn,"wc_output":train_labels_wc,"wo_output":train_labels_wo},
		validation_data=(#[test_input_ids,test_input_masks,test_segment_ids, test_weights],
					[test_input_ids,test_input_masks,test_segment_ids, test_tok_mask, test_head_mask, test_head_col_mask],
						 {"sc_output":test_labels_sc,"sa_output":test_labels_sa,
						  "wn_output":test_labels_wn,"wc_output":test_labels_wc,
						 "wo_output":test_labels_wo}),
		epochs=1,
		batch_size=32
	)
		
	
	x = sad

	# In[68]:


	i = 1
	test_example = test_data[i]
	p_id,p_mask,p_seg = formatForBert([test_example])

	input1,input2,input3,input4 = p_id,p_mask,p_seg,np.array([test_weights[i,:]])


	# In[70]:


	test_example


	# In[69]:


	model.predict([input1,input2,input3,input4])


	# In[89]:


	{"sc_output":train_labels_sc,"sa_output":train_labels_sa,
	 "wn_output":train_labels_wn,"wc_output":train_labels_wc,"wo_output":train_labels_wo},
	train_labels_wo.max()


	# In[143]:


	


	# In[19]:


	wc.shape,wo.shape

