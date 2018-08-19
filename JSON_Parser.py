import pandas as pd
import json
json_data = json.load(open('demo.json'))

len(json_data)
Time = []
GENRE_COL = []
PROB = []


#for i in range(len(json_data)):
#    Time.append(json_data[i][0])

for i in range(len(json_data)):
    Time.extend([json_data[i][0] for j in range(10)])

for t in range(len(json_data)):
    GENRE_COL.extend(list(json_data[t][1].keys()))
    PROB.extend(list(json_data[t][1].values()))
    
len(Time)
len(GENRE_COL)
len(PROB)


data_tuples = list(zip(Time,GENRE_COL,PROB))
data_tuples
df = pd.DataFrame(data_tuples)

groups = df.groupby(pd.cut(df.index, range(0,len(df), 10)))
print(groups.max())




df.to_csv('example_lz.csv')