from operator import concat
import pandas as pd
import config
import os
from utils import order

useful_cols = ['model', 
                'heads',
                'data',
                'lr',
                'encoder-feature-layers',
                'accuracy_val',
                'me_accuracy_val',
                'f1_val',
                'me_f1_val',
                'recall_val',
                'me_recall_val',
                'precision_val',
                'me_precision_val',
                'f1_macro_val',
                'me_f1_macro_val',
                'recall_macro_val',
                'me_recall_macro_val',
                'precision_macro_val',
                'me_precision_macro_val',
                'f1_weighted_val',
                'me_f1_weighted_val',
                'recall_weighted_val',
                'me_recall_weighted_val',
                'precision_weighted_val',
                'me_precision_weighted_val']

new_cols = {'accuracy_val':'acc',
            'me_accuracy_val':'me_acc',
            'f1_val':'f1',
            'me_f1_val':'me_f1',
            'recall_val':'recall',
            'me_recall_val':'me_recall',
            'precision_val':'precision', 
            'me_precision_val':'me_precision',
            'f1_macro_val':'f1_macro',
            'me_f1_macro_val':'me_f1_macro',
            'recall_macro_val':'recall_macro',
            'me_recall_macro_val':'me_recall_macro',
            'precision_macro_val':'precision_macro',
            'me_precision_macro_val':'me_precision_macro',
            'f1_weighted_val':'f1_weighted',
            'me_f1_weighted_val':'me_f1_weighted',
            'recall_weighted_val':'recall_weighted',
            'me_recall_weighted_val':'me_recall_weighted',
            'precision_weighted_val':'precision_weighted',
            'me_precision_weighted_val':'me_precision_weighted'
            }

#load data
models_dir = [d for d in os.listdir(config.RESULTS_PATH) if os.path.isdir(config.RESULTS_PATH+'/'+d)]

# read grid search data from models
model_results = [pd.read_csv(config.RESULTS_PATH + '/' + model + '/' + config.DOMAIN_GRID_SEARCH + '.csv', usecols=useful_cols) for model in models_dir]

# rename columns
for df in model_results:
    df.rename(columns=new_cols, inplace=True)

# concat dataframes
concat_results = pd.concat(model_results).reset_index(drop=True)

# data info
specific_data = {'EXIST':{'df':None, 'metric':'acc'}, 'HatEval':{'df':None, 'metric':'f1_macro'}, 'DETOXIS':{'df':None, 'metric':'f1'}}

# select rows belonging to certain data/task
for name in specific_data.keys():
    specific_data[name]['df'] = concat_results.loc[concat_results['data'] == name]

# group by model and heads based on a metric
for name in specific_data.keys():
    # specific_data[name]['df'] = specific_data[name]['df'].groupby(['model','heads','data'], sort=False).max(specific_data[name]['metric']
    specific_data[name]['df'] = specific_data[name]['df'].loc[specific_data[name]['df'].groupby(['model','heads','data'])[specific_data[name]['metric']].idxmax()]
    # remove the data column
    specific_data[name]['df'].drop('data', axis=1, inplace=True)

# Merge dataframes from diferent tasks/data
df_merge = pd.merge(specific_data['EXIST']['df'], specific_data['DETOXIS']['df'], how='outer', on=['model','heads'])
df_merge = pd.merge(df_merge, specific_data['HatEval']['df'], how='outer', on=['model','heads'])

# apply function to order models list
models = order(df_merge['model'].unique())

# building the final table
models_data=[]
for model in models:
    # get model rows
    # df_specific_model = df_merge.iloc[df_merge.index.get_level_values('model') == model]
    df_specific_model = df_merge.loc[df_merge['model'] == model]
    # sort column headds by length
    s = df_specific_model.heads.str.len().sort_values().index
    # reindex the table by heads column
    models_data.append(df_specific_model.reindex(s))


# concatenate all models data
df_all = pd.concat(models_data).reset_index(inplace=False, drop=True)

# save final table
print(df_all.head(11))
df_all.to_csv(config.RESULTS_PATH + '/' + 'final.csv', index=False)


