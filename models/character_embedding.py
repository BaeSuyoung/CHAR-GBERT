from utils import *
import argparse
import sys

'''
config
'''
parser = argparse.ArgumentParser()
parser.add_argument("--vector_size", type=int, default=768)
parser.add_argument("--w2v_epochs", type=int, default=60)
parser.add_argument("--window", type=int, default=1)
parser.add_argument("--min_count", type=int, default=1)
parser.add_argument("--sg", type=int, default=1)
parser.add_argument("--workers", type=int, default=4)
parser.add_argument("--seed", type=int, default=1212)
parser.add_argument("--mode", type=str, default="train")

args = parser.parse_args()
graph_type_list= ["w2v", "avg", "moral_without", "moral_with"]

cfg_vector_size=args.vector_size # 768
cfg_w2v_epochs=args.w2v_epochs # 60
cfg_window=args.window
cfg_min_count=args.min_count
cfg_sg=args.sg
cfg_workers=args.workers
cfg_seed=args.seed
cfg_mode = args.mode

f=pd.read_csv("../data/story_dataset/datainfo.csv", sep="   ")
for index, row in f.iterrows():
    print(row['num'], row['data_type'], row['file_name'], row['main_char'])
    data_type=row['data_type']
    file_name=row['file_name']
    main_char=row['main_char'][1:-1]
    main_char=main_char.split(",")
    word_special_token=[]
    for i in range(len(main_char)):
        word_special_token.append(f'[unused{i}]')

    dataset_path=f"../data/story_dataset/clean/{file_name}_clean_mfd.csv"
    dataset=pd.read_csv(dataset_path)
    model_folder=f"../outputs/graph_models/{file_name}"
    createFolder(model_folder)

    for graph_type in graph_type_list:
        if graph_type== "moral_without" or graph_type== "moral_with":
            graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{graph_type}.model"
            dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{graph_type}.pickle"
            model=co_training(dataset, word_special_token, graph_type, cfg_seed, cfg_vector_size, cfg_window, cfg_min_count, cfg_workers, cfg_w2v_epochs)
            model.save(graph_path) # all

            # character embedding
            character_embeddings= make_character_embeddings(model, False)
            with open(dict_graph_path,'wb') as fw:
                pickle.dump(character_embeddings, fw)

        elif graph_type=="w2v":
            graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{graph_type}.model"
            dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{graph_type}.pickle"
            model=w2v_training(dataset, cfg_vector_size,cfg_window, cfg_min_count, cfg_sg, cfg_workers, cfg_w2v_epochs)
            model.save(graph_path)
            character_embedding= make_character_embeddings(model, True)

            with open(dict_graph_path,'wb') as fw:
                pickle.dump(character_embedding, fw)

        elif graph_type=="avg":
            graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_w2v.model"
            dict_graph_path=f"../outputs/graph_models/{file_name}/{file_name}_{cfg_vector_size}_{graph_type}.pickle"
            character_embedding=avg_emb(dataset, word_special_token, graph_path) 
            
            with open(dict_graph_path,'wb') as fw:
                pickle.dump(character_embedding, fw)