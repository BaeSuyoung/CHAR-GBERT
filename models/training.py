from transformers import AutoTokenizer
from transformers.models.bert.modeling_bert import *
from transformers.modeling_utils import *
from CharBertModel import *
from utils import *
import argparse
import sys


import pandas as pd
import numpy as np
import random
import pickle
import torch
from tqdm import tqdm
import time
from torch import nn
from transformers import AdamW
import os
import torch
from tqdm import tqdm
import time
from torch import nn
from transformers import AdamW
from keras.utils import to_categorical
from sklearn import preprocessing


'''
config
'''
parser = argparse.ArgumentParser()
parser.add_argument("--vector_size", type=int, default=768)
parser.add_argument("--ep", type=int, default=30)
parser.add_argument("--filter_mfd", type=bool, default=False)
parser.add_argument("--seq_len", type=int, default=200)
parser.add_argument("--bs", type=int, default=50)
parser.add_argument("--lr", type=int, default=0.00001)
parser.add_argument("--seed", type=int, default=1212)
parser.add_argument("--model", type=str, default="bert-base-uncased")

args = parser.parse_args()

cfg_vector_size=args.vector_size # 768
cfg_train_epochs=args.ep
cfg_sentence_length=args.seq_len
cfg_filter_mfd=False
cfg_batch_size=args.bs
cfg_learning_rate=args.lr
cfg_seed=args.seed
cfg_baseline_model=args.model


deterministic = True
random.seed(cfg_seed)
np.random.seed(cfg_seed)
torch.manual_seed(cfg_seed)
torch.cuda.manual_seed_all(cfg_seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def preprocess_dataset(dataset, is_filter):
    df=dataset[["chapter", "mask_sent", "char", "mfd2", "mfd_word"]]
    if is_filter:
        df=df[df["mfd2"]!="X"]
    df["sentence_mfd"]=df["mask_sent"]+"[sep]"+df["mfd_word"]
    df=df.reset_index(drop=True)
    return df

def dataloader(tokenizer, df, max_seq_length, batch_size):
    train_inputs = tokenizer(list(df["sentence_mfd"]), return_tensors='pt', max_length=max_seq_length, truncation=True, padding='max_length')
    #print(train_inputs.input_ids)
    label = preprocessing.LabelEncoder()
    train_y = label.fit_transform(df['char'])
    train_y = to_categorical(train_y)
    train_inputs['labels']=train_y
    voc_size=tokenizer.vocab_size

    for i in range(train_inputs.input_ids.shape[0]):
        mask_arr=(train_inputs.input_ids[i]==tokenizer.convert_tokens_to_ids(df["char"][i]))
        train_inputs.input_ids[i, torch.flatten(mask_arr.nonzero()).tolist()] = 103

    train_dataset = MeditationsDataset(train_inputs)
    train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

    return train_dataloader

def train(model, train_dataloader, character_embeddings, device, learning_rate, train_epochs, character_length, vector_size, model_save_path):
    prev_save_step=-1
    # sentiment==1
    model.to(device)
    optimizer = AdamW(model.parameters(), lr=learning_rate)
    train_start = time.time()
    ## training ##
    for epoch in range(train_epochs):
        loop = tqdm(train_dataloader, leave=True)
        model.train()
        optimizer.zero_grad()

        for step, batch in enumerate(loop):
            if prev_save_step >-1:
                if step<=prev_save_step: continue
            if prev_save_step >-1:
                prev_save_step=-1

            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            labels=batch['labels'].to(device)

            if character_embeddings !=None:
                character_embedding=character_embeddings.expand(len(input_ids), character_length, vector_size).to(device)
                outputs = model(character_embedding, input_ids, attention_mask=attention_mask) #logits
            else:
                outputs = model(character_embedding=None, input_ids=input_ids, attention_mask=attention_mask) #logits
            #print(outputs)
            loss_fct = nn.BCEWithLogitsLoss()
            loss = loss_fct(outputs, labels)

            loss.backward()

            optimizer.step()
            optimizer.zero_grad()

            loop.set_description(f'Epoch {epoch}')
            loop.set_postfix(loss=loss.item())

            if step % 1000 == 0:
                print("Epoch:{}/{}, Train {} Loss: {}, Cumulated time: {}m ".format(epoch, step, len(train_dataloader), loss.item(),(time.time() - train_start)/60.0))

        print('--------------------------------------------------------------')
        
        to_save={'epoch': epoch,
                'model_state': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': loss,
                'lower_case': True}
        torch.save(to_save, model_save_path)
    return model

model_type_list= ["without", "w2v", "avg", "moral_without", "moral_with"]

f=pd.read_csv("../data/story_dataset/datainfo.csv", sep="   ")
for index, row in f.iterrows():
    torch.cuda.empty_cache()
    print(row['num'], row['data_type'], row['file_name'], row['main_char'])
    data_type=row['data_type']
    file_name=row['file_name']
    main_char=row['main_char'][1:-1]
    main_char=main_char.split(",")
    word_special_token=[]
    for i in range(len(main_char)):
        word_special_token.append(f'[unused{i}]')

    cfg_character_length=len(main_char)
    cfg_max_seq_length = cfg_character_length+cfg_sentence_length

    dataset_path=f"../data/story_dataset/clean/{file_name}_clean_mfd.csv"
    dataset=pd.read_csv(dataset_path)
    train_df=preprocess_dataset(dataset, cfg_filter_mfd)

    tokenizer = AutoTokenizer.from_pretrained(cfg_baseline_model)
    model = CharBertModel.from_pretrained(cfg_baseline_model, cfg_character_length, cfg_max_seq_length, cfg_character_length)

    train_dataloader=dataloader(tokenizer, train_df, cfg_max_seq_length, cfg_batch_size)
    torch.cuda.empty_cache()
    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

    print(f"training num: {len(train_df)}")
    print(f"base model: {cfg_baseline_model}")
    print(f"device: {device}")

    for model_type in model_type_list:
        print(f"model type: {model_type}")
        model_folder=f"../outputs/story_models/{file_name}"
        createFolder(model_folder)
        model_save_path=f"../outputs/story_models/{file_name}/{file_name}_{model_type}_{cfg_vector_size}_{cfg_character_length}.pt"

        if model_type=="without":
            character_embeddings=None
            train_model=train(model, train_dataloader, None, device, cfg_learning_rate, cfg_train_epochs, cfg_character_length, cfg_vector_size, model_save_path)  
        else:
            dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{model_type}.pickle"
            with open(dict_graph_path, 'rb') as fr:
                user_loaded = pickle.load(fr)
            #make character embedding
            tmp=[]
            for key in user_loaded.keys():
                tmp.append(torch.Tensor(user_loaded[key]))
            character_embeddings=torch.stack(tmp, 0)
            # model trainin            
            train_model=train(model, train_dataloader, character_embeddings, device, cfg_learning_rate, cfg_train_epochs, cfg_character_length, cfg_vector_size, model_save_path)
