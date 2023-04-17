import pandas as pd
import config
import os


path = config.REPO_PATH + '/results/' + 'mtl2'
file_name = 'gridsearch.csv'

dir_list = [d for d in os.listdir(path) if os.path.isdir(path+'/'+d)]
df_list = [pd.read_csv(path + '/' + d + '/' + file_name) for d in dir_list]
df = pd.concat(df_list)

# save final table
print(df.head(11))
df.to_csv(path + '/' + file_name, index=False)