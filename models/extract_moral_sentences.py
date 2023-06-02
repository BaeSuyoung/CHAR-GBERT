from utils import *
import argparse
import sys

from datasets import load_dataset

# config
parser = argparse.ArgumentParser()
parser.add_argument("--ds", type=str, default="story")
args = parser.parse_args()
cfg_ds = args.ds

dataset_list={"moral_stories","story", "overall", "mft"}

if cfg_ds not in dataset_list:
    sys.exit("Dataset choice error!")
# mfd dictionary
mfd, lemma_words=mfd2_make_dict()
# one_moral_dataset making
if cfg_ds=="moral_stories":
    df = load_dataset('demelin/moral_stories', 'full', split='train')
    moral_dataset=evaluate_dataset_preprocessing(df)
    moral_dataset["char_num"].value_counts()
    moral_dataset=moral_dataset[(moral_dataset["char_num"]=='[1,1]')]
    moral_dataset.to_csv("../data/moral_dataset/one_moral_dataset.csv")

if cfg_ds=="overall":
    print("Extracting overall evaluation dataset")
    dataset=pd.read_csv("../data/moral_dataset/one_moral_dataset.csv")
    dataset=dataset[["moral_action", "immoral_action", "situation_moral", "situation_immoral"]]
    sample_df=extract_mfd_sentence_overall(dataset, mfd, lemma_words, cfg_ds)
    sample_df.to_csv("../data/moral_dataset/moral_df_101.csv")

if cfg_ds=="mft":
    print("Extracting mft evaluation dataset")
    dataset=pd.read_csv("../data/moral_dataset/one_moral_dataset.csv")
    dataset=dataset[["moral_action", "immoral_action", "situation_moral", "situation_immoral"]]
    sample_df=extract_mfd_sentence_overall(dataset, mfd, lemma_words, cfg_ds)
    sample_df.to_csv("../data/moral_dataset/moral_df_mfd2.0_3792.csv")

if cfg_ds=="story":
    print("Extracting story datasets")
    f=pd.read_csv("../data/story_dataset/datainfo.csv", sep="   ")
    for index, row in f.iterrows():
        print(row['num'], row['data_type'], row['file_name'], row['main_char'])
        data_type=row['data_type']
        file_name=row['file_name']
        main_char=row['main_char'][1:-1]
        main_char=main_char.split(",")

        dataset_path=f"../data/story_dataset/clean/{file_name}_clean.txt"
        output_path=f"../data/story_dataset/clean/{file_name}_clean_mfd.csv"
        df_train=pd.read_csv(dataset_path)
        sample_df=extract_mfd_sentence_story(df_train, mfd, lemma_words)
        sample_df.to_csv(output_path)
        print(f"training_num: {len(sample_df)}")
