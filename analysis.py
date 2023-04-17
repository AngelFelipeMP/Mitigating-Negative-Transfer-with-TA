import numpy as np
import pandas as pd
from transformers import AutoTokenizer
import matplotlib.pyplot as plt
from config import *
import re

if __name__ == "__main__":
    # create a list with merged files
    merged_files = [file for file in os.listdir(DATA_PATH) if 'merge' in file]

    for file in merged_files:
        print('\nTokens Analyses: {}'.format(file.split('_')[0]))
        print('File: {}'.format(file))
        df = pd.read_csv(DATA_PATH + '/' + file)

        for transformer in TRANSFORMERS:
            
            print(f'Transformer: {transformer} \n')
            tokenizer = AutoTokenizer.from_pretrained(transformer)
            
            # tokenize text
            df['tokens'] = df[INFO_DATA[re.sub(r'\d+', '',file.split('_')[0])]['text_col']].apply(lambda x: tokenizer.tokenize(x))
            # count tokens
            df['number_tokens'] = df['tokens'].apply(lambda x: len(x))

            #print a histogram of tokens
            ax = df['number_tokens'].plot.hist(bins=20, range=(0, 512))
            ax.plot()
            plt.show()

            # print % of tokens covered by each max_len
            print('Percentage of comments under 64 tokens: ',((df.query('number_tokens < 64').size / df.size) * 100))
            print('Percentage of comments under 128 tokens: ',((df.query('number_tokens < 128').size / df.size) * 100))
            print('Percentage of comments under 256 tokens: ',((df.query('number_tokens < 256').size / df.size) * 100))
            print('Percentage of comments under 512 tokens: ',((df.query('number_tokens < 512').size / df.size) * 100))
        
        
        
        
        


























    # df_train = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_TRAIN, sep='\t')
    # df_dev = pd.read_csv(config.DATA_PATH + '/' + config.DATASET_DEV, sep='\t')
    # df = pd.concat([df_train, df_dev]).reset_index(drop=True)

    # for transformer in config.TRANSFORMERS:
        
    #     print(f'Transformer: {transformer} \n')

    #     tokenizer = AutoTokenizer.from_pretrained(transformer)

    #     df['tokens'] = df['text_processed'].apply(lambda x: tokenizer.tokenize(x))
    #     df['number_tokens'] = df['tokens'].apply(lambda x: len(x))

    #     ax = df['number_tokens'].plot.hist(bins=20, range=(0, 512))
    #     ax.plot()
    #     plt.show()

    #     print('Porcentage of comments under 64 tokens: ',((df.query('number_tokens < 64').size / df.size) * 100))
    #     print('Porcentage of comments under 128 tokens: ',((df.query('number_tokens < 128').size / df.size) * 100))
    #     print('Porcentage of comments under 256 tokens: ',((df.query('number_tokens < 256').size / df.size) * 100))
    #     print('Porcentage of comments under 512 tokens: ',((df.query('number_tokens < 512').size / df.size) * 100))