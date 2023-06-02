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
import torch
import torch.nn.functional as F
import pandas as pd
import random
from sklearn.preprocessing import normalize

'''
config
'''
parser = argparse.ArgumentParser()
parser.add_argument("--eval", type=str, default="overall")
parser.add_argument("--sm", type=str, default="softmax")
parser.add_argument("--nm", type=str, default="none")
parser.add_argument("--vector_size", type=int, default=768)
parser.add_argument("--seq_len", type=int, default=200)
parser.add_argument("--lr", type=int, default=0.00001)
parser.add_argument("--seed", type=int, default=1212)
parser.add_argument("--model", type=str, default="bert-base-uncased")

model_type_list= ["without", "w2v", "avg", "moral_without", "moral_with"]

args = parser.parse_args()
eval_type={"overall", "mft"}
sorting_mode={"softmax", "score"}
norm_mode={"none", "norm"}
cfg_eval_type=args.eval
cfg_sorting_mode=args.sm # softmax, score
cfg_norm_mode=args.nm # norm, none

if cfg_eval_type not in eval_type:
    sys.exit("evluation type error!")

if cfg_sorting_mode not in sorting_mode:
    sys.exit("sorting mode type error!")

if cfg_norm_mode not in norm_mode:
    sys.exit("norm mode type error!")

cfg_vector_size=args.vector_size # 768
cfg_sentence_length=args.seq_len
cfg_learning_rate=args.lr
cfg_seed=args.seed
cfg_baseline_model=args.model
cfg_sorting_mode=args.sm # softmax, score
cfg_norm_mode=args.nm # norm, none


deterministic = True
random.seed(cfg_seed)
np.random.seed(cfg_seed)
torch.manual_seed(cfg_seed)
torch.cuda.manual_seed_all(cfg_seed)
if deterministic:
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def softmax(x):
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True) *100

def evaluation(tokenizer, model, max_length, character_length, vector_size, sent, sort, norm, special_token, character_embedding, device):
    device='cpu'
    model.eval()
    result=[]
    for i in range(len(sent)):
        sent[i]=sent[i].replace("CHAR_0", "[MASK]")
        input_tokens = tokenizer(sent[i], return_tensors='pt', max_length=max_length, truncation=True, padding='max_length')["input_ids"] # moral

        with torch.no_grad():
            if character_embedding !=None:
                character_embedding=character_embedding.expand(len(input_tokens), character_length, vector_size)
                character_embedding=character_embedding.to(device)
                input_tokens=input_tokens.to(device)
                #print(character_embedding.get_device(), input_tokens.get_device())
                logits = model(character_embedding.to(device), input_tokens.to(device)) #logits
            else:
                logits = model(character_embedding=None, input_ids=input_tokens.to("cpu")) #logits
            logit=logits.split(len(special_token), dim=1)[0]
            if norm=="norm":
                logit=F.normalize(logit, dim=1)
            if sort=="softmax":
                logit=F.softmax(logit, dim=1)

            result.append(logit[0].cpu().numpy())
    df=pd.DataFrame(result, index=sent, columns=special_token)
    df=df.fillna(0)
    return df

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

    tokenizer = AutoTokenizer.from_pretrained(cfg_baseline_model)
    model = CharBertModel.from_pretrained(cfg_baseline_model, cfg_character_length, cfg_max_seq_length, cfg_character_length)

    story_evaluation_result=[]
    if cfg_eval_type=="overall":
        morality_dataset=pd.read_csv("../data/moral_dataset/moral_df_101.csv")
        for random in [1,2,3,4,5]:
            morality_dataset=morality_dataset.sample(n=100, random_state=random)
            moral=list(morality_dataset["moral_action"]) 
            immoral=list(morality_dataset["immoral_action"])

            model_results=[]
            for model_type in model_type_list:
                print(f"random: {random}, model type: {model_type}")
                model_save_path=f"../outputs/story_models/{file_name}/{file_name}_{model_type}_{cfg_vector_size}_{cfg_character_length}.pt"

                optimizer = AdamW(model.parameters(), lr=cfg_learning_rate)
                checkpoint = torch.load(model_save_path)
                model.load_state_dict(checkpoint['model_state'])
                optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                epoch = checkpoint['epoch']
                loss = checkpoint['loss']

                if model_type=="without":
                    character_embeddings=None
                else:
                    dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{model_type}.pickle"
                    with open(dict_graph_path, 'rb') as fr:
                        user_loaded = pickle.load(fr)
                    tmp=[]
                    for key in user_loaded.keys():
                        tmp.append(torch.Tensor(user_loaded[key]))
                    character_embeddings=torch.stack(tmp, 0)

                torch.cuda.is_available()
                torch.cuda.empty_cache()
                device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

                eval_moral_result=evaluation(tokenizer, model, max_length=cfg_max_seq_length, 
                                                character_length=cfg_character_length, 
                                                vector_size=cfg_vector_size, 
                                                sent=moral, sort=sorting_mode, norm=norm_mode, 
                                                special_token=main_char, 
                                                character_embedding=character_embeddings, 
                                                device=device)
                eval_immoral_result=evaluation(tokenizer, model, max_length=cfg_max_seq_length, 
                                                character_length=cfg_character_length, 
                                                vector_size=cfg_vector_size, 
                                                sent=immoral, sort=sorting_mode, norm=norm_mode, 
                                                special_token=main_char, 
                                                character_embedding=character_embeddings, 
                                                device=device)
                eval_morality_result=pd.DataFrame()
                result=[]
                for item in eval_moral_result.columns:
                    eval_morality_result[item]=list((eval_moral_result[item].values)-(eval_immoral_result[item].values))
                    char_sum=len(eval_morality_result[eval_morality_result[item]>0])
                    result.append(char_sum) # 1*3
                if len(result)==2:
                    result.append(0)
                model_results.append(result) # 5*3
            story_evaluation_result.append(model_results) # 3 * 5 * 3
        print(np.mean(np.array(story_evaluation_result), axis=0))


    elif cfg_eval_type=="mft":
        morality_dataset=pd.read_csv("../data/moral_dataset/moral_df_mfd2.0_3792.csv")
        moral_list=[(100,['care.virtue', 'care.vice']), (50,['fairness.virtue', 'fairness.vice']),(5,['loyalty.virtue', 'loyalty.vice']),(100,['authority.virtue', 'authority.vice']),(100,['sanctity.virtue', 'sanctity.vice'])]
        for i, mft_list in moral_list:
            morality_result=[]
            for random in [20,21,22,23,24]:
                moral=list(morality_dataset[(morality_dataset["type"]==mft_list[0])]["action"].sample(i, random_state=random))
                immoral=list(morality_dataset[(morality_dataset["type"]==mft_list[1])]["action"].sample(i, random_state=random))
                model_results=[]
                for model_type in model_type_list:
                    print(f"mft: {mft_list}, seed: {random}, model type: {model_type}")
                    model_save_path=f"../outputs/story_models/{file_name}/{file_name}_{model_type}_{cfg_vector_size}_{cfg_character_length}.pt"

                    optimizer = AdamW(model.parameters(), lr=cfg_learning_rate)
                    checkpoint = torch.load(model_save_path)
                    model.load_state_dict(checkpoint['model_state'])
                    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
                    epoch = checkpoint['epoch']
                    loss = checkpoint['loss']

                    if model_type=="without":
                        character_embeddings=None
                    else:
                        dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{model_type}.pickle"
                        with open(dict_graph_path, 'rb') as fr:
                            user_loaded = pickle.load(fr)
                        tmp=[]
                        for key in user_loaded.keys():
                            tmp.append(torch.Tensor(user_loaded[key]))
                        character_embeddings=torch.stack(tmp, 0)

                    torch.cuda.is_available()
                    torch.cuda.empty_cache()
                    device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

                    eval_moral_result=evaluation(tokenizer, model, max_length=cfg_max_seq_length, 
                                                    character_length=cfg_character_length, 
                                                    vector_size=cfg_vector_size, 
                                                    sent=moral, sort=sorting_mode, norm=norm_mode, 
                                                    special_token=main_char, 
                                                    character_embedding=character_embeddings, 
                                                    device=device)
                    eval_immoral_result=evaluation(tokenizer, model, max_length=cfg_max_seq_length, 
                                                    character_length=cfg_character_length, 
                                                    vector_size=cfg_vector_size, 
                                                    sent=immoral, sort=sorting_mode, norm=norm_mode, 
                                                    special_token=main_char, 
                                                    character_embedding=character_embeddings, 
                                                    device=device)
                    eval_morality_result=pd.DataFrame()
                    result=[]
                    for item in eval_moral_result.columns:
                        eval_morality_result[item]=list((eval_moral_result[item].values)-(eval_immoral_result[item].values))
                        char_sum=len(eval_morality_result[eval_morality_result[item]>0])
                        result.append(char_sum) # 1*3
                    if len(result)==2:
                        result.append(0) # 
                    model_results.append(result) # model *3
                morality_result.append(model_results) # random * model * char
            print(f"story: {file_name}, {mft_list} score in each model is")
            print(softmax(normalize(np.mean(np.array(morality_result), axis=0)))) # model * char
