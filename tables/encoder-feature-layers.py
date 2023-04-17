from operator import concat
import pandas as pd
import config
import os

useful_cols = ['model', 
                'heads',
                'encoder-feature-layers',
                'data',
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

new_cols = {'encoder-feature-layers':'layers',
            'accuracy_val':'acc',
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

# get grid mtl2 files
path_mtl2 = os.path.join(config.RESULTS_PATH, 'mtl2')
model_files = [d for d in os.listdir(path_mtl2) if os.path.isfile(path_mtl2 + '/' + d) and 'pred' not in d]

# read grid search data from model mtl2
model_results = [pd.read_csv(path_mtl2 + '/' + file , usecols=useful_cols) for file in model_files]

# rename columns
for df in model_results:
    df.rename(columns=new_cols, inplace=True)

# concat dataframes
if len(model_results) > 1:
    concat_results = pd.concat(model_results).reset_index(drop=True)
else:
    concat_results = model_results[0]

# remove model column
concat_results.drop('model', axis=1, inplace=True)

# data info
specific_data = {'EXIST':{'df':None, 'metric':'acc'}, 'HatEval':{'df':None, 'metric':'f1_macro'}, 'DETOXIS':{'df':None, 'metric':'f1'}}

for name in specific_data.keys():
    # select rows belonging to certain data/task
    specific_data[name]['df'] = concat_results.loc[concat_results['data'] == name]
    # group heads by average/max
    # specific_data[name]['df'] = specific_data[name]['df'].groupby(['heads','layers','data'], sort=False).mean() #COMMENT: I must try max and mean
    specific_data[name]['df'] = specific_data[name]['df'].loc[specific_data[name]['df'].groupby(['heads','layers','data'])[specific_data[name]['metric']].idxmax()]
    # remove the data column
    specific_data[name]['df'].drop('data', axis=1, inplace=True)
    
    print(specific_data[name]['df'])     

# Merge dataframes from diferent tasks/data
# df_merge = pd.merge(specific_data['EXIST']['df'], specific_data['DETOXIS']['df'], how='outer', on=['model','heads'])
# df_merge = pd.merge(df_merge, specific_data['HatEval']['df'], how='outer', on=['model','heads'])
df_merge = pd.merge(specific_data['EXIST']['df'], specific_data['DETOXIS']['df'], how='outer', on=['heads','layers'])
df_merge = pd.merge(df_merge, specific_data['HatEval']['df'], how='outer', on=['heads','layers'])

print(df_merge)

# sort column heads by length
s = df_merge.heads.str.len().sort_values().index
# reindex the table by heads column
df_merge.reindex(s)

# save final table
print(df_merge.head(11))
df_merge.to_csv(config.RESULTS_PATH + '/' + 'encoder-feature-layers.csv', index=False)