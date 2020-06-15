'''
This file uses twarc API to fetch COVID19 data from tweets
Due to volume of data, we used stratified sampling

@copyright Lun Li, Shengjie Liu
'''

# Import Packages
import os
import csv
import pandas as pd
from twarc import Twarc

#/Users/lunli/Documents/COVID-19-TweetIDs-master/
PATH = '2020-04/' # choose a particular month
OUTPUT_PATH = 'tweet_new/'
LOG_FILE = 'Logs/logs.csv'

# 60000 per day
# hard-coded
weights = dict()
weights['04'] = 1200; weights['05'] = 1200
weights['06'] = 1200; weights['07'] = 1200
weights['08'] = 1200; weights['09'] = 1200
weights['10'] = 1200; weights['11'] = 1200
weights['00'] = 3150; weights['01'] = 3150
weights['02'] = 3150; weights['03'] = 3150
weights['12'] = 3150; weights['13'] = 3150
weights['14'] = 3150; weights['15'] = 3150
weights['16'] = 3150; weights['17'] = 3150
weights['18'] = 3150; weights['19'] = 3150
weights['20'] = 3150; weights['21'] = 3150
weights['22'] = 3150; weights['23'] = 3150


def sample_file(fileName, numSamples):
    df = pd.read_csv(fileName, names = ["ids"])
    numSamples = len(df) if len(df) < numSamples else numSamples
    ids = df['ids'].sample(n = numSamples, random_state = 1)
    return ids.values
    

# Use Twarc extract covid19 related tweets
twarc = Twarc()
tmp_df = pd.read_csv(LOG_FILE, names = ["file"])
traversed = list(tmp_df.file.values)
with open(LOG_FILE, 'a+') as logf:
    for file in os.listdir(PATH):
        if file not in traversed:
            file_postfix = str(file).split(".")[0][-2:]
            sample_size = weights[file_postfix]
            print("Extract from file: ", file, "for ", sample_size, " samples:")
            ids = sample_file(PATH + file ,sample_size)
            output_file_name = str(file).split(".")[0] + "_contents.txt"
            # log
            w_ = csv.writer(logf); w_.writerow([file])
            # extract content
            with open(OUTPUT_PATH + output_file_name, 'w') as wf:                
                for tweet in twarc.hydrate(ids):
                    if "retweeted_status" in tweet:
                        if tweet['lang'] != "en":
                            continue
                        else:
                            row = []
                            row.append(tweet['created_at'])
                            row.append(tweet['id_str'])
                            row.append(tweet['retweeted_status']['full_text'])
                            w = csv.writer(wf, delimiter = ',')
                            w.writerow(row)
        else:
            print(file, " is processed already!")

