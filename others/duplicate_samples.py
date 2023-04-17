from config import *

import pandas as pd
for task in INFO_DATA.keys():
    
    print('###################################################')
    print(task)
    df  = pd.read_csv(DATA_PATH + '/' + str(INFO_DATA[task]['datasets']['test'].split('.')[0]) + '_processed.csv')
    print(df[df.duplicated(subset=[INFO_DATA[task]['text_col']], keep=False)])
    

