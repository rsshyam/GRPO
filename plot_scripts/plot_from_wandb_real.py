import wandb
import numpy as np
import matplotlib.pyplot as plt
from scipy.stats import sem
from visualisations_utils_wandb_api import (
    download_runs,
    process_max_fields,
    process_runs,
    group_process_runs
    )
from collections import defaultdict
import os
import neatplot
neatplot.set_style()


# Constants and configurations
ENTITY = 'robust-rl-project'
PROJECT = 'group-robust-dpo-neurips'
dataset_group='goqa'

#group=f'{dataset_group}_{group_indices}'+f'tr_frac{config.train_frac}'+f'{config.model.name_or_path}_spairs_{config.sep_pairs}_{config.trainer}',
            
SETTINGS = {
    'goqa': ['goqa_0_1tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer','goqa_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer'],#goqa_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer
    'goqma': ['goqma_2tr_frac0.8google/gemma-2b_spairs_False_GroupTrainer']
}
ALGORITHMS = {
    'rdpo': 'Robust DPO',
    'dpo': 'DPO',
    'sft': 'SFT',
    'base': 'Base_Model'
}

def get_setting_details(setting_key: str):
    if 'all' in setting_key:
        pass
    group_list = SETTINGS[setting_key]
    #weights_array = np.array(group_list[0].split('weights[')[-1].split(']')[0].split(','), dtype=float)
    #pref_data_num = group_list[0].split('pref_data_num')[1].split('weights')[0]
    return group_list#, weights_array, pref_data_num

def create_filter_dicts(groups: list[str],n_epochs: int, base: bool=False):
    base_filter_dpo = {
        'config.loss.name': 'dpo',
        'State': 'finished',
        'config.n_epochs': n_epochs
    }
    base_filter_rdpo = {
        'State': 'finished',
        'config.loss.name': 'rdpo',
        'config.n_epochs': n_epochs
    }
    base_filter_sft = {
        'State': 'finished',
        'config.loss.name':'sft',
        'config.n_epochs': 1
    }
    base_filter_gemma_base = {
        'State': 'finished',
        'config.loss.name':'base'
    }

    #if len(groups)==1: # IPO
    rdpo_filter = {**base_filter_rdpo, 'group': groups[0], 'config.loss.importance_sampling': False, 'config.loss.step_size': 0.05 }
    rdpo_filter_2 = {**base_filter_rdpo, 'group': groups[1], 'config.loss.importance_sampling': False, 'config.loss.step_size': 0.1 }
    dpo_filter = {**base_filter_dpo, 'group': groups[1]}
    sft_filter = {**base_filter_sft, 'group': groups[0]}
    gemma_base_filter={**base_filter_gemma_base, 'group': groups[0]}
    return [rdpo_filter, rdpo_filter_2, dpo_filter,gemma_base_filter] if base else [rdpo_filter, dpo_filter,  rdpo_filter_2]

    filters = []
    for group in groups:
        filter = {
            **base_filter_dpo, 
            'group': group, 
            'config.ipo_grad_type': 'justdpo',
            'config.dpo_type': 'dpo' if 'dpo' in group else 'rdpo', 
            'config.importance_sampling': 'imp' in group,
            'config.importance_sampling_weights': {'$nin': ['0.5,0.5']}, 
            'config.use_theory': False
        }
        filters.append(filter)
    return filters

def determine_algorithm(filters_dict):
    if filters_dict['config.loss.name'] == 'rdpo': # IPO
        if filters_dict['config.loss.importance_sampling'] == True:
            return 'DPO Importance Sampling'
        step_size=filters_dict['config.loss.step_size']
        return f'RDPO_{step_size}'
    
    if filters_dict['config.loss.name'] == 'dpo':
        return 'DPO'
    return 'RDPO'

def prepare_metric_data(filters_dicts, metrics,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations,metric_titles):
    metric_values = []
    metric_sem = []
    labels = []
    for metric_name in metrics:
        for i,filters_dict in enumerate(filters_dicts):
            algo = filters_dict['config.loss.name']
            avg = all_avg_metrics_at_iterations[metric_name][i]
            sem = all_sem_metrics_at_iterations[metric_name][i]
            #name=metric_titles[metric_name]
            metric_values.append(avg)
            metric_sem.append(sem)
            labels.append(algo)
    return metric_values, metric_sem, labels

def plot_metric_with_error_bands(iteration_index, metric_values, metric_sem, labels, plot_title, subfolder_path, file_name,metric, colors=None, extend=False):
    plt.figure(figsize=(12, 6))
    #for i, (avg, sem) in enumerate(zip(metric_values, metric_sem)):
    for avg, sem, label in zip(metric_values, metric_sem, labels):
        if extend and len(avg) != len(iteration_index):
            avg = np.append(avg, [avg[-1]] * (len(iteration_index) - len(avg)))
            sem = np.append(sem, [sem[-1]] * (len(iteration_index) - len(sem)))
        #color = colors[i] if colors else None
        plt.plot(iteration_index, avg, label=label)
        plt.fill_between(iteration_index, avg - sem, avg + sem, alpha=0.2)
    plt.title(plot_title,fontsize=40)
    plt.xlabel('Iterations',fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=40)
    safe_title = file_name.replace('/', '-')
    neatplot.save_figure(f'{subfolder_path}/{safe_title}')
    plt.close()

def plot_metric_bars_dpo(metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations):
    plt.figure(figsize=(12, 6))

    for i, filters_dict in enumerate(filters_dicts):
        algo = determine_algorithm(filters_dict)
        metrics_end_avg = [all_avg_metrics_at_iterations[metric][i][-1] for metric in metric_config['metrics']]
        metrics_end_sem = [all_sem_metrics_at_iterations[metric][i][-1] for metric in metric_config['metrics']]
        
        bar_width = 0.1 if 'group_loss' in metric_config['metrics'][0] else 0.2
        offset = i * bar_width
        positions = np.arange(len(metrics_end_avg)) + offset
        
        plt.bar(positions, height=metrics_end_avg, yerr=metrics_end_sem, width=bar_width, capsize=5, alpha=0.7, label=f'{algo}')
        plt.xticks(positions, [f"Group {i+1}" for i in range(len(metrics_end_avg))],fontsize=40)

    plt.title(metric_config['title'],fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=40)
    safe_title = metric_config["title"].replace('/', '-')
    neatplot.save_figure(f'{subfolder_path}/{safe_title}')
    plt.close()
    # Define bar properties

def plot_metric_bars(metric_config, filters_dicts, subfolder_path, all_avg_metrics_at_iterations, all_sem_metrics_at_iterations):
    plt.figure(figsize=(12, 6))
    print(metric_config)

    color_map = {
        'base': 'blue',   # Blue for 'base'
        'sft': 'green',   # Green for 'sft'
        'rdpo': 'red',    # Red for 'rdpo'
        'dpo': 'purple'   # Purple for 'dpo'
    }

    # Define bar properties
    num_groups = 2
    num_bars_per_group = 4  # Adjust as needed per group
    bar_width = 0.2
    group_width = num_bars_per_group * bar_width + 0.1  # Adjust spacing between groups
    for i, filters_dict in enumerate(filters_dicts):
        algo = filters_dict['config.loss.name']
        # Initialize lists to store metrics for plotting
        groups = [[], []]  # Two groups for each type of metric

        # Gather data for each metric
        for metric in metric_config['metrics']:
            index = int(metric.split('_')[-1])  # This assumes metrics end in '_0' or '_1'
            category = ''
            if 'logps_accuracies_eval' in metric:
                category = 'base'
            elif 'logps_ref_eval/accuracies' in metric:
                category = 'sft'
            elif 'logps_pol_eval/accuracies' in metric:
                if 'rdpo' in algo:
                    category = 'rdpo' 
                    category_nan = "dpo"
                else:
                    category = 'dpo' 
                    category_nan = "rdpo"
                groups[index].append((category_nan,np.nan,np.nan))
            else:
                if 'rdpo' in algo:
                    category = 'rdpo' 
                    category_nan = "dpo"
                else:
                    category = 'dpo' 
                    category_nan = "rdpo"
                groups[index].append((category_nan,np.nan,np.nan))
            # Collect data
            avg = all_avg_metrics_at_iterations[metric][i][-1]
            sem = all_sem_metrics_at_iterations[metric][i][-1]
            groups[index].append((category, avg, sem))

        # Sort groups by predefined order if necessary
        order = ['base', 'sft', 'dpo', 'rdpo']
        #print(groups)
        groups = [sorted(g, key=lambda x: order.index(x[0])) for g in groups]
        #print(groups)
        # Plot each group
        for group_index, group_data in enumerate(groups):
            for bar_index, data in enumerate(group_data):
                #print(data)
                position = group_index * group_width + bar_index * bar_width
                plt.bar(position, height=data[1], yerr=data[2], width=bar_width, capsize=5,color=color_map[data[0]], alpha=0.7, label=f'{data[0]}' if i == 0 and group_index == 0 else "")

    plt.xticks([group_width/2 + i * group_width for i in range(num_groups)], ['Group 0', 'Group 1'], fontsize=20)
    plt.title(metric_config['title'],fontsize=40)
    plt.ylabel('Value',fontsize=40)
    plt.legend(fontsize=30)
    neatplot.save_figure(f'{subfolder_path}/{metric_config["file_suffix"]}')
    plt.close()

def generate_metrics(base_name, count, mode='eval', separator='_'):
    metrics = []
    if '/' in base_name:
        # Split the base name on the slash
        parts = base_name.split('/')
        # Reconstruct the base name with the mode inserted before the slash
        modified_base_name = f'{parts[0]}{separator}{mode}/{parts[1]}'
    else:
        # If no slash, proceed normally
        modified_base_name = f'{base_name}{separator}{mode}'

    # Generate metric names based on the modified base name
    for i in range(count):
        metrics.append(f'{modified_base_name}_{i}')
    
    return metrics
    return metrics

def main():
    setting = 'goqa'  # convention X_Y_Z: X={'even','uneven'}, Y={'balanced','imbalanced'}, Z={'dpo','ipo','all'}
    n_epochs=5
    group_count=2
    groups= get_setting_details(setting)
    filters_dicts = create_filter_dicts(groups,n_epochs)
    
    #metrics_to_collect = ['grad_norm', 'train_loss', 'reward_err_1', 'reward_err_2', 'reward_param_1', 'reward_param_2', 'reward_param_3', 'reward_param_4','group_weight_1','group_weight_2','val_loss','train_group_loss_1','train_group_loss_2','val_group_loss_1','val_group_loss_2','hist_group_loss_1','hist_group_loss_2','max_val_grp_loss','max_train_grp_loss','max_reward_err','max_kl_dist']
    #metrics_to_collect = ['logps_accuracies_eval_0','logps_accuracies_eval_1','logps_pol_eval/accuracies_0','logps_pol_eval/accuracies_1','logps_ref_eval/accuracies_0','logps_ref_eval/accuracies_1','loss/train_0','loss/train_1','loss/eval_0','loss/eval_1']
    

    # Define the configuration for each metric group
    metric_configurations = [
        {'base_name': 'logps/chosen', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps/rejected', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps/chosen', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps/rejected', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_pol/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_pol/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps_ref/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'logps_ref/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/chosen', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/rejected', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/chosen', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/rejected', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'rewards/margins', 'count': group_count, 'mode': 'train', 'separator': '_'},
        {'base_name': 'rewards/margins', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'loss', 'count': group_count, 'mode': 'train', 'separator': '/'},
        {'base_name': 'loss', 'count': group_count, 'mode': 'eval', 'separator': '/'},
        {'base_name': 'logps_accuracies', 'count': group_count, 'mode': 'eval', 'separator': '_'},
        {'base_name': 'logps_accuracies', 'count': group_count, 'mode': 'train', 'separator': '_'},
    ]

    # Initialize an empty list to collect all generated metrics
    metrics_list = []

    # Generate metrics for each configuration and add them to the list
    for config in metric_configurations:
        generated_metrics = generate_metrics(
            base_name=config['base_name'],
            count=config['count'],
            mode=config['mode'],
            separator=config['separator']
        )
        metrics_list.extend(generated_metrics)

    # Print the full list of generated metrics
    
    metrics_list.append('loss/train')
    metrics_to_collect=metrics_list
    print(metrics_to_collect)
    
    
    
    all_metrics_history = {metric: [] for metric in metrics_to_collect}

    all_runs=[]
    processed_filters = []
    # Loop through each filters_dict value
    for filters_dict in filters_dicts:
        # Download runs for the current filters_dict
        runs = download_runs(ENTITY, PROJECT, filters_dict)
        if len(runs)>0:
            all_runs.append(runs)
            print(len(runs))
            metrics_history = {}

            for metric in metrics_to_collect:
                metrics_history[metric] = process_runs(runs, field=metric, time_field='_step')

            # Accumulate metrics data for each configuration
            for metric in metrics_to_collect:
                all_metrics_history[metric].append(metrics_history[metric])
            processed_filters.append(filters_dict)
    #print(all_metrics_history)
    filters_dict=processed_filters
    iteration_len=0
    iteration_index=0
    for runs in all_runs:
        for run in runs:
            iteration_index_1=run[['_step']].dropna().values.ravel()
            #print(iteration_index_1)
            if len(iteration_index_1)>iteration_len:
                iteration_len=len(iteration_index_1)
                iteration_index=iteration_index_1
    print(iteration_index,iteration_len)

    base_folder = f'wandb-plots-gemma/{len(filters_dicts)}_setting_{setting}'
    os.makedirs(base_folder, exist_ok=True)
    subfolder_name = f"{filters_dicts[0]['config.loss.name']}{len(filters_dicts)}"
    subfolder_path = os.path.join(base_folder, subfolder_name)
    os.makedirs(subfolder_path, exist_ok=True)

    all_avg_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}
    all_sem_metrics_at_iterations = {metric: [] for metric in metrics_to_collect}

    for i, filters_dict in enumerate(filters_dicts):
        for metric in metrics_to_collect:
            values_matrix = all_metrics_history[metric][i]
            #print('VAL MATRIX: ', values_matrix[0:2])
            #print(values_matrix)
            avg_values = np.mean(values_matrix, axis=0)
            sem_values = sem(all_metrics_history[metric][i], axis=0)
            all_avg_metrics_at_iterations[metric].append(avg_values.ravel())
            all_sem_metrics_at_iterations[metric].append(sem_values.ravel())

    # Create a default dictionary to hold the grouped metrics
    plot_configs = defaultdict(list)

    # Parse each metric and group by a derived key (ignoring the numeric suffix)
    for metric in metrics_to_collect:
        key_parts = metric.rsplit('_', 1)[0]  # Split off the numeric suffix
        plot_configs[key_parts].append(metric)  # Append metric to its group

    # Convert defaultdict to a regular dict
    plot_configs_dict = dict(plot_configs)
    
    print(plot_configs_dict)
    titles_dict = {
        'logps_accuracies': 'Log Likelihood Accuracies (Chosen vs Rejected)',
        'loss/train': 'Group Train Loss',
        'loss/eval': 'Group Validation Loss'
    }
    
    metrics_titles = {
        'logps_accuracies_eval_0' : 'Log Likelihood Accuracies Group-0',
        'logps_accuracies_eval_1' : 'Log Likelihood Accuracies Group-1'
    }
    


    for metric, metrics in plot_configs_dict.items():
        values, sems, labels = prepare_metric_data(filters_dicts, metrics,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations,metrics)
        #metric_name = "_".join(metrics)
        #title=titles_dict[metric]
        plot_metric_with_error_bands(iteration_index, values, sems, labels, f'{metric} over Iterations', subfolder_path, f"{metric}", metric, extend=True)
    # Define a list of metric configurations for each plot
    #metrics_configs = [
    #    {'metrics': [metric for metric in metrics_to_collect if 'logps' in metric], 'title': 'Log Likelihood Accuracies (Chosen vs Rejected)', 'file_suffix': 'log_accuracy'},
    #    {'metrics': [metric for metric in metrics_to_collect if 'loss/train' in metric], 'title': 'Group Train Loss at the End', 'file_suffix': 'train_group_loss_bars'},
    #    {'metrics': [metric for metric in metrics_to_collect if 'loss/eval' in metric], 'title': 'Group Validation Loss at the End', 'file_suffix': 'val_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_reward_err' in metric], 'title': 'Max Reward Error at the End', 'file_suffix': 'max_reward_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_train_grp_loss' in metric], 'title': 'Max Group Train Loss at the End', 'file_suffix': 'max_train_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_val_grp_loss' in metric], 'title': 'Max Group Validation Loss at the End', 'file_suffix': 'max_val_group_loss_bars'},
        #{'metrics': [metric for metric in metrics_to_collect if 'max_kl_dist' in metric], 'title': 'Max KL Distance at the End', 'file_suffix': 'max_kl_distance_bars'}
    #]
    #print(metrics_configs)
    # Loop through each configuration and plot
    for metric,metrics in plot_configs_dict.items():
        plot_metric_bars_dpo({'title': metric, 'metrics':metrics}, filters_dicts, subfolder_path,all_avg_metrics_at_iterations,all_sem_metrics_at_iterations)

if __name__ == "__main__":
    main()


   



#group=f'{dataset_group}_{group_indices}'+f'tr_frac{config.train_frac}'+f'{config.model.name_or_path}_spairs_{config.sep_pairs}_{config.trainer}',
            