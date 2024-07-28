
import datasets
import torch
import json
from torch.utils.data import DataLoader, Dataset
from src.utils import get_local_dir, TemporarilySeededRandom
from torch.nn.utils.rnn import pad_sequence
from collections import defaultdict
import tqdm
import random
from bs4 import BeautifulSoup, NavigableString
import numpy as np
from typing import Dict, List, Optional, Iterator, Callable, Union, Tuple
import pandas as pd
import ast
import matplotlib.pyplot as plt
from typing import Literal

COUNTRIES=[
    'Nigeria', 
    'Egypt', 
    'India (Current national sample)', 
    'China', 
    'Japan', 
    'Germany', 
    'France', 
    'Spain', 
    'United States', 
    'Canada', 
    'Brazil', 
    'Argentina', 
    'Australia', 
    'New Zealand'
]


def load_and_prepare_data(dataset_name: str, split: str, group_filter: List[str], cache_dir: str = None):
    try:
        # Try to load the dataset from the Hugging Face Hub
        dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    except Exception as e:
        print(f"Error loading dataset from Hugging Face Hub: {e}")
        print("Loading dataset from local storage...")
        dataset = datasets.load_dataset("/home/uceesr4/Group-robust-preference-optimization/llm_global_opinions")["train"]
    #dataset = datasets.load_dataset(dataset_name, cache_dir=cache_dir)["train"]
    #print(dataset)                                                                       
    df = pd.DataFrame(dataset)
    df['qkey'] = df.index

    new_selections = []
    new_rows = []
    new_options = []
    for i in range(len(df)):
        if not df.loc[i, "question"] or not df.loc[i, "options"]:
            continue
        selections_str = "{" + df.loc[i, "selections"].split("{")[1].split("}")[0] + "}"
        selections_dict = ast.literal_eval(selections_str)
        #print(selections_dict,group_filter)
        if group_filter and not any(country in selections_dict for country in group_filter):
            #for country in group_filter:
            #    if not selections_dict[country] or len(selections_dict[country])==0 or np.sum(selections_dict[country]) == 0:
            continue
        
        ##one condition missing, need to be checked later
        new_selections.append(selections_dict)
        new_rows.append(df.loc[i])
        parsed_options = ast.literal_eval(df.loc[i, "options"])
        new_options.append([str(opt) for opt in parsed_options])

    return pd.DataFrame(new_rows), new_selections, new_options

def process_data_frame(df, selections,group_filter, options):
    df['selections'] = selections
    df['options'] = options
    df['selections'] = df['selections'].apply(lambda x: [(k, v) for k, v in x.items()])  # create country - selections tuples
    df = df.explode('selections', ignore_index=True)
    #print(df['selections'])
    df[['group', 'prob_y']] = pd.DataFrame(df['selections'].tolist(), index=df.index)
    df = df[df['prob_y'].apply(lambda x: x is not None and len(x) > 0 and np.sum(x) > 0)]
    return df[df['group'].isin(group_filter)]

def plot_questions_by_country(df, title_suffix=""):
    count = df['group'].value_counts()
    count.plot(kind='bar', figsize=(10, 6), color='skyblue')
    plt.title(f'Number of Questions by Country {title_suffix}')
    plt.xlabel('Country')
    plt.ylabel('Number of Questions')
    plt.xticks(rotation=45, ha='right')
    plt.savefig(f'questions_by_country{title_suffix}.png')
    plt.close()

def create_goqa_data(df,split,train_frac=0.8, multi_pair=False,n_pairs=4):
    df_train=df.sample(frac=0.8,random_state=42)
    df_test=df.drop(df_train.index)

    df_truetest=df_test.sample(frac=0.5,random_state=42)
    df_valtest=df_test.drop(df_truetest.index)
    if split=='train':
        if train_frac < 0.8:
            df=df_train.sample(frac=(train_frac/0.8),random_state=42)
        else:
            df=df_train
    elif split=='test':
        df=df_test
    elif split=='truetest':
        df=df_truetest
    elif split=='valtest':
        df=df_valtest
    else:
        print(split)
        raise Exception('incorrect split')
    grouped = df.groupby('group')
    data = defaultdict(lambda: defaultdict(list))
    for group_name, group_data in grouped:
        for qkey, sub_group in group_data.groupby('qkey'):

            question = sub_group['question'].values[0]

            # Process options, excluding any invalid ones
            options = sub_group['options'].values[0]
            #####treat refused option separately-----options = [opt for opt in options if opt != "Refused"]

            # Construct the prompt
            prompt = f"Opinion of people in {group_name} on: {question}\nPlease select the best response:"

            # Generate the full prompt with options
            letters = 'ABCDEFGHIJKLMNOPQRSTUVWXYZ'[:len(options)]
            for opt, letter in zip(options, letters):
                prompt += f"\n{letter}. {opt}"
            #print(sub_group)
            responses = [letter for letter in letters[:len(options)]]
            
            prob_y = torch.tensor(np.stack(sub_group['prob_y'].values), dtype=torch.float).squeeze()
            ranks=torch.argsort(prob_y)
            pairs = [(ranks[i], ranks[j]) for i in range(len(ranks)) for j in range(i)]
            correct_response_index = ranks[-1]
            correct_response = responses[ranks[-1]]

            data[prompt]['sft_target'] = correct_response
            data[prompt]['responses'] = responses
            if multi_pair:
                data[prompt]['pairs']=random.sample(pairs,min(n_pairs,len(pairs)))
            else:
                wrong_indices = [i for i in range(len(options)) if i != correct_response_index]
                if wrong_indices:
                    wrong_response_index = random.choice(wrong_indices)
                    data[prompt]['pairs']=[(correct_response_index,wrong_response_index)]
            #print(data[prompt])
    #print(len(data))
    return data

def get_goqa(split: str, train_frac: float = 0.8, group_id: int = None, multi_pair: bool = False,n_pairs: int=4, silent: bool = False, cache_dir: str = None):
    if group_id==None:
        group_filter = COUNTRIES
    else:
        group_filter = [COUNTRIES[group_id]]
    df, selections, options = load_and_prepare_data("Anthropic/llm_global_opinions", split, group_filter, cache_dir)
    df = process_data_frame(df, selections, group_filter, options)
    plot_questions_by_country(df, title_suffix=f" {split} with groups {' '.join(group_filter)}")
    #print(group_id,group_filter)
    return create_goqa_data(df=df,split=split,train_frac=train_frac,multi_pair= multi_pair,n_pairs=n_pairs)



##test
# Example usage:
#data_train = get_goqa("train", group_filter=["USA", "UK"])
#data_test = get_goqa("test", multi_response=True)
#print(data_train)
#print(data_test)


def create_goqa_data_alt(
        df: pd.DataFrame, 
        split: Literal['train','test'], 
        train_frac: float = 0.8, 
        multi_response: bool = False, 
        option_mode: Literal['preferred_least_min_gap','balanced'] | None = None,
    ) -> dict[list]:
    # Sampling train and test sets
    df_train = df.sample(frac=0.8, random_state=42)
    df_test = df.drop(df_train.index)

    # Selecting the data split
    if split == 'train':
        if train_frac < 0.8:
            df = df_train.sample(frac=(train_frac / 0.8), random_state=42)
        else:
            df = df_train
    elif split == 'test':
        df = df_test
    else:
        raise ValueError('incorrect split value')

    grouped = df.groupby('group')
    data = defaultdict(lambda: defaultdict(list))

    if option_mode == 'preferred_least_min_gap':
        group_sizes = grouped.size()
        sorted_groups = group_sizes.sort_values()
        cumulative_sizes = sorted_groups.cumsum()
        half_data_point = cumulative_sizes.iloc[-1] / 2
        closest_split_index = (cumulative_sizes - half_data_point).abs().argmin()
        preferred_groups = set(sorted_groups.iloc[:closest_split_index + 1].index)
        minimal_gap_groups = set(sorted_groups.iloc[closest_split_index + 1:].index)

    # Process each group
    for group_name, group_data in grouped:
        for qkey, sub_group in group_data.groupby('qkey'):
            prob_y = torch.tensor(np.stack(sub_group['prob_y'].values), dtype=torch.float).squeeze()
            prompt = f"{sub_group['questions'].values[0]} Opinion of people in country: {group_name} is"
            options = sub_group['options'].values[0]

            if option_mode == 'preferred_least_min_gap':
                if group_name in preferred_groups:
                    # Most and least preferred selection
                    max_index = torch.argmax(prob_y)
                    min_index = torch.argmin(prob_y)
                    selected_indices = [max_index, min_index]
                else:
                    # Minimal gap selection
                    sorted_indices = np.argsort(prob_y)
                    min_gap = float('inf')
                    selected_pair = (0, 1)
                    for i in range(len(prob_y) - 1):
                        gap = prob_y[sorted_indices[i+1]] - prob_y[sorted_indices[i]]
                        if gap < min_gap:
                            min_gap = gap
                            selected_pair = (sorted_indices[i], sorted_indices[i+1])
                    selected_indices = list(selected_pair)
            elif option_mode == 'balanced':
                selected_indices = np.random.choice(len(options), 2, replace=False)
            else:
                # Default behavior or other modes can be added here
                max_index = torch.argmax(prob_y)
                selected_indices = [max_index, np.random.choice([i for i in range(len(options)) if i != max_index])]

            # Adding the correct and selected responses
            for index in selected_indices:
                if multi_response:
                    data[prompt]['responses'].append(options[index])
                else:
                    data[prompt]['pairs'].append((len(data[prompt]['responses']), len(data[prompt]['responses']) + 1))
                    data[prompt]['responses'].extend([options[selected_indices[0]], options[index]])
                if multi_response:
                    for i, option in enumerate(sub_group['options'].values[0]):
                        if i != correct_response_index:
                            data[prompt]['pairs'].append((len(data[prompt]['responses']), len(data[prompt]['responses'])+1))
                            data[prompt]['responses'].extend([correct_response, option])
                else:
                    wrong_indices = [i for i in range(len(sub_group['options'].values[0])) if i != correct_response_index]
                    if wrong_indices:
                        wrong_option_index = np.random.choice(wrong_indices)
                        wrong_response = sub_group['options'].values[0][wrong_option_index]
                        data[prompt]['pairs'].append((len(data[prompt]['responses']), len(data[prompt]['responses'])+1))
                        data[prompt]['responses'].extend([correct_response, wrong_response])
    return data