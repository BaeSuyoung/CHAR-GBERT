import pandas as pd
import nltk
from nltk.stem import WordNetLemmatizer
import spacy
from nltk.corpus import stopwords

from stellargraph.data import BiasedRandomWalk
from stellargraph import StellarGraph
from gensim.models import Word2Vec
from nltk.tokenize import word_tokenize

import networkx as nx
import os
import json
import csv
import pickle
from collections import Counter
import re
import numpy as np



NER=spacy.load('en_core_web_sm')
nltk.download('wordnet')
lemma=WordNetLemmatizer()

def createFolder(directory):
    try:
        if not os.path.exists(directory):
            os.makedirs(directory)
    except OSError:
        print ('Error: Creating directory. ' +  directory)

'''
Lemmatization
input : text (str)
output : lemma text (str)
'''
def lemmatize(text):
    doc=NER(text)
    text = " ".join([token.lemma_ for token in doc])
    return text


'''
Remove stop words
input : text (str)
output : removed text (str)
'''
def remove_stop_words(text):
    stop_words = set(stopwords.words('english'))
    tokens=[tok for tok in text.split(" ")]
    clean_tokens=[tok for tok in tokens if len(tok)>1 and (tok.lower() not in stop_words)]
    text=" ".join(clean_tokens)
    return text


'''
Estimation: Extract character in the text
input : text (str)
output : replaced text , character num
'''
def character_extractions(text):
    ner_list= NER(text)
    char=set()
    for e in ner_list.ents:
        if e.label_ =='PERSON':
            char.add(e.text)
    if len(char) != 0:
        char=list(char)
        for c in range(len(char)):
            text=text.replace(char[c], f"CHAR_{c}")
    return text, len(char)


def preprocess(text):
    # remove stop words
    text=remove_stop_words(text)
    # lemmatize
    text=lemmatize(text)
    # extract characters, character numberts
    text, char_num=character_extractions(text)
    return text, char_num

'''
Moral Dataset Preprocessing
'''
def evaluate_dataset_preprocessing(df):
    moral_dataset=pd.DataFrame()
    moral_action=[]
    immoral_action=[]
    char_num=[]
    for i in range(len(df)):
        # moral, immoral action ner-tagging
        text, moral_char_num=preprocess(df["moral_action"][i])
        moral_action.append(text)

        text, immoral_char_num=preprocess(df["immoral_action"][i])
        immoral_action.append(text)

        char_num.append(f"[{moral_char_num},{immoral_char_num}]")

    moral_dataset['moral_action'], moral_dataset['immoral_action'],moral_dataset['char_num'] = moral_action, immoral_action, char_num
    moral_dataset['situation']=df["situation"]
    moral_dataset['norm']=df["norm"]
    return moral_dataset


'''
Make MFD 2.0 dictionary
'''
def mfd2_make_dict():    
    MFD2 = '../data/moral_dataset/mfd2.0.dic'
    nummap = dict()
    mfd2 = dict()
    wordmode = True
    with open(MFD2, 'r') as f:
        for line in f.readlines():
            ent = line.strip().split()
            if line[0] == '%':
                wordmode = not wordmode
            elif len(ent) > 0:
                if wordmode:
                    wordkey = ''.join([e for e in ent if e not in nummap.keys()])
                    mfd2[wordkey] = [nummap[e] for e in ent if e in nummap.keys()]
                else:
                    nummap[ent[0]] = ent[1]

    mfd2 = pd.DataFrame.from_dict(mfd2).T
    mfd2_foundations = mfd2[0].unique()
    mfd2['foundation'] = mfd2[0]
    del mfd2[0]
    mfd2 = mfd2.T.to_dict()

    lemma_words=[lemma.lemmatize(w) for w in list(mfd2.keys())]
    return mfd2, lemma_words

'''
Story dataset : extract mfd sentence
'''
def extract_mfd_sentence_story(dataset, mfd, lemma_words):
    mfd2, mfd2_word=[], []
    mfd2_list=list(mfd.keys())
    for i in range(len(dataset)):
        mfd_list=[]
        mfd_words=[]
        sentence_list=dataset["text"][i].split(" ")
        for i in range(len(mfd2_list)):
            if mfd2_list[i] in sentence_list or lemma_words[i] in sentence_list:
                mfd_list.append(mfd[mfd2_list[i]]["foundation"])
                mfd_words.append(mfd2_list[i])
        mfd_list=list(set(mfd_list))
        mfd_words=list(set(mfd_words))
        if len(mfd_list)==0:
            mfd2.append("X")
            mfd2_word.append("X")
        else:
            mfd2.append(",".join(mfd_list))
            mfd2_word.append(",".join(mfd_words))

    dataset["mfd2"], dataset["mfd_word"]=mfd2, mfd2_word    
    dataset=dataset[["chapter", "text", "mask_sent", "char", "mfd2", "mfd_word"]]
    return dataset

'''
Moral stories dataset : extract mfd sentence
'''
def extract_mfd_sentence_overall(dataset, mfd, lemma_words, ds):
    moral_mfd=[]
    moral_mfd_word=[]
    immoral_mfd=[]
    immoral_mfd_word=[]
    mfd2_list=list(mfd.keys())
    for i in range(len(dataset)):
        moral_mfd_list=[]
        moral_mfd_words=[]
        immoral_mfd_list=[]
        immoral_mfd_words=[]
        moral_sentence_list=dataset["moral_action"][i].split(" ")
        immoral_sentence_list=dataset["immoral_action"][i].split(" ")
        for i in range(len(mfd2_list)):
            if mfd2_list[i] in moral_sentence_list or lemma_words[i] in moral_sentence_list:
                moral_mfd_list.append(mfd[mfd2_list[i]]["foundation"])
                moral_mfd_words.append(mfd2_list[i])
            if mfd2_list[i] in immoral_sentence_list or lemma_words[i] in immoral_sentence_list:
                immoral_mfd_list.append(mfd[mfd2_list[i]]["foundation"])
                immoral_mfd_words.append(mfd2_list[i])
        moral_mfd_list=list(set(moral_mfd_list))
        moral_mfd_words=list(set(moral_mfd_words))
        immoral_mfd_list=list(set(immoral_mfd_list))
        immoral_mfd_words=list(set(immoral_mfd_words))

        if len(moral_mfd_list)==0:
            moral_mfd.append("X")
            moral_mfd_word.append("X")
        else:
            moral_mfd.append(",".join(moral_mfd_list))
            moral_mfd_word.append(",".join(moral_mfd_words))

        if len(immoral_mfd_list)==0:
            immoral_mfd.append("X")
            immoral_mfd_word.append("X")
        else:
            immoral_mfd.append(",".join(immoral_mfd_list))
            immoral_mfd_word.append(",".join(immoral_mfd_words))

    dataset["moral_mfd2"], dataset["moral_mfd_word"]=moral_mfd, moral_mfd_word    
    dataset["immoral_mfd2"], dataset["immoral_mfd_word"]=immoral_mfd, immoral_mfd_word    
    
    if ds=="overall":
        dataset=dataset[(dataset["moral_mfd2"]!="X") & (dataset["immoral_mfd2"]!="X")]
        dataset=dataset[(dataset["moral_mfd2"].str.contains("virtue")) & (dataset["immoral_mfd2"].str.contains("vice"))]
        dataset=dataset[(~dataset["moral_mfd2"].str.contains("vice")) & (~dataset["immoral_mfd2"].str.contains("virtue"))]
        dataset=dataset.reset_index(drop=True)
        dataset=dataset[["moral_action", "immoral_action",  "situation_moral", "situation_immoral", "moral_mfd2", "immoral_mfd2", "moral_mfd_word", "immoral_mfd_word"]]
        #dataset=dataset[dataset["mfd2"]!="X"]
        print(f"overall evaluation dataset : {len(dataset)}")
        return dataset
    
    elif ds=="mft":
        mfd_dict={
            "care.virtue":[],
            "care.vice":[],
            "fairness.virtue":[],
            "fairness.vice":[],
            "loyalty.virtue":[],
            "loyalty.vice":[],
            "authority.virtue":[],
            "authority.vice":[],
            "sanctity.virtue":[],
            "sanctity.vice":[],
        }

        for i in range(len(dataset)):
            for key in mfd_dict:
                if (key in dataset["moral_mfd2"][i]) and ("virtue" in dataset["moral_mfd2"][i]):
                    mfd_dict[key].append(dataset["moral_action"][i])
                if (key in dataset["immoral_mfd2"][i]) and ("vice" in dataset["immoral_mfd2"][i]):
                    mfd_dict[key].append(dataset["immoral_action"][i])
        for key, val in zip(mfd_dict.keys(), mfd_dict.values()):
            print(key, ":", len(val))
        t_type=[]
        t_action=[]
        for key, val in zip(mfd_dict.keys(), mfd_dict.values()):
            t_type.extend([key]*(len(val)))
            t_action.extend(val)

        df=pd.DataFrame()
        df['type'], df['action']=t_type, t_action
        print(f"mft evaluation dataset : {len(df)}")
        return df

''' 
dictionary: (sentence_id : character occurrence list)
'''
def section_dict(sentences, character_tokens):
    sections_dictionary = {}
    iterative = 0
    for sentence in sentences:
        iterative += 1
        for char in character_tokens:
            if char in sentence:
                if str(iterative) in sections_dictionary.keys():
                    sections_dictionary[str(iterative)].append(char)  
                else:
                    sections_dictionary[str(iterative)] = [char]    

    return sections_dictionary

''' 
co-occurrence assiciated with character sentiment
'''
def co_occurrence(sections_dictionary, character_tokens, moral_sentence_labels, moral_network):
    df = pd.DataFrame(columns = character_tokens, index = character_tokens)
    df[:] = int(0)

    for idx, value in zip(sections_dictionary.keys(), sections_dictionary.values()):
        for i in range(len(value)):
            df[value[i]][value[i]]+=1
        
        for character1 in value:
            for character2 in value:
                if character1 != character2:
                    if moral_network=="moral_with":
                        mfd2_list=moral_sentence_labels[int(idx)-1].split(",")
                        for mfd2 in mfd2_list:
                            if "vice" in mfd2: # negative
                                df[character1][character2] += 1
                            elif "virtue" in mfd2: #positive
                                df[character1][character2] += 3
                            else:  #x
                                df[character1][character2] += 2
                    else:
                        df[character1][character2] += 2

    return df

def make_graph(df, character_tokens):
    edge_list = [] #test networkx
    for index, row in df.iterrows():
        i = 0
        for col in row:
            weight = float(col)
            edge_list.append((index, df.columns[i], weight))
            i += 1
    #Remove edge if 0.0
    updated_edge_list = [x for x in edge_list if not x[2] == 0.0]

    #create duple of char, occurance in novel
    node_list = []
    for i in character_tokens:
        for e in updated_edge_list:
            if i == e[0] and i == e[1]:
                node_list.append((i, e[2]))
            
    for i in node_list:
        if i[1] == 0.0:
            node_list.remove(i)

    #remove self references
    for i in updated_edge_list:
        if i[0] == i[1]:
            updated_edge_list.remove(i)

    return updated_edge_list, node_list

#networkx graph time!
def add_graph(updated_edge_list, node_list):
    G = nx.Graph()
    for i in sorted(node_list):
        G.add_node(i[0], size = i[1])
    G.add_weighted_edges_from(updated_edge_list)
    return G


#Couldn't determine another route to listing out the order of nodes for future work
def updated_node_order(G, node_list):
    node_order = nx.nodes(G)
    nodes = []
    for i in node_order:
        for x in node_list:
            if x[0] == i:
                nodes.append(x)
    return nodes

#reorder edge list
def update_edges(G):
    test = nx.get_edge_attributes(G, 'weight')
    updated_again_edges = []
    for i in nx.edges(G):
        for x in test.keys():
            if i[0] == x[0] and i[1] == x[1]:
                updated_again_edges.append(test[x])
    return updated_again_edges


# weighted node2vec training
def get_weighted_walks(G, walk_length, n, p, q, weighted, seed):
    walk_length = walk_length  # maximum length of a random walk to use throughout this notebook
    Graph = StellarGraph.from_networkx(G)
    #print(Graph.info())

    rw = BiasedRandomWalk(Graph)
    weighted_walks = rw.run(
        nodes=Graph.nodes(),  # root nodes
        length=walk_length,  # maximum length of a random walk
        n=n,  # number of random walks per root node
        p=p,  # Defines (unormalised) probability, 1/p, of returning to source node
        q=q,  # Defines (unormalised) probability, 1/q, for moving away from source node
        weighted=weighted,  # for weighted random walks
        seed=seed,  # random seed fixed for reproducibility
    )
    #print("Number of random walks: {}".format(len(weighted_walks)))
    return weighted_walks

# word2vec model training
def word2vec_training(G, weighted_walks, vector_size, window, min_count, workers, epochs):
    weighted_model = Word2Vec(sentences=weighted_walks, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    weighted_model.build_vocab(corpus_iterable=[list(G.nodes())])
    weighted_model.train(weighted_walks, total_examples=weighted_model.corpus_count, epochs=epochs)
    return weighted_model

# story dataset w2v embedding dict
def make_character_embeddings(model, flag):
    w2v_id_list=model.wv.index_to_key
    character_embedding={}
    for id in w2v_id_list:
        if "unused" in id:
            if flag:
                idx="["+id+"]"
            else: idx=id
            character_embedding[idx]=model.wv[id]
    return character_embedding

def co_training(dataset, word_special_token, graph_type, seed, vector_size, window, min_count, workers, epochs):
    moral_sentence_labels=list(dataset['mfd2']) # moral labels
    sentences=list(dataset['mask_sent']) # sentences
    sections_dictionary=section_dict(sentences, word_special_token)
    df=co_occurrence(sections_dictionary, word_special_token, moral_sentence_labels, graph_type)
    updated_edge_list, node_list=make_graph(df, word_special_token)
    G=add_graph(updated_edge_list, node_list)

    updated_nodes=updated_node_order(G, node_list)
    updated_edges=update_edges(G)    

    weighted_walks=get_weighted_walks(G, walk_length=100, n=50, p=0.5, q=2.0, weighted=True, seed=seed)

    weighted_model = Word2Vec(sentences=weighted_walks, vector_size=vector_size, window=window, min_count=min_count, workers=workers)
    weighted_model.build_vocab(corpus_iterable=[list(G.nodes())])
    weighted_model.train(weighted_walks, total_examples=weighted_model.corpus_count, epochs=epochs)

    return weighted_model

def w2v_training(dataset, vector_size, window, min_count, sg, workers, epochs):
    sentences=dataset["mask_sent"]
    tokenize_texts = [word_tokenize(sentence) for sentence in sentences]
    vocab=[]
    for text in tokenize_texts:
        vocab.extend(text)
    vocab=set(vocab)

    model = Word2Vec(vector_size=vector_size, window=window, min_count=min_count, sg=sg, workers=workers, epochs=epochs)
    model.build_vocab(corpus_iterable=[list(vocab)])
    model.train(tokenize_texts, total_examples=model.corpus_total_words, epochs=epochs)

    return model

def avg_emb(dataset, character_info, graph_path):
    character_embedding={}
    model=Word2Vec.load(graph_path)

    for character in character_info:
        datasets=dataset[dataset["char"]==character]
        if len(datasets)>1:
            tokenize_texts = [word_tokenize(sentence) for sentence in datasets["mask_sent"]]
            vocab=[]
            for text in tokenize_texts:
                vocab.extend(text)
            vocab=set(vocab)

            node_ids=[]
            for word in list(vocab):
                for i in model.wv.index_to_key:
                    if i==word:
                        node_ids.append(i)
            node_embeddings = (
                    model.wv[node_ids]
                )
            out=node_embeddings
            character_embedding[character]=np.mean(out, axis=0) 
    return character_embedding


