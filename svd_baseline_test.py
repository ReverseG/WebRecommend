import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from svd_baseline import SVD


record_num = 100000
part_data = pd.read_csv('data/men/MenUI.csv')[:record_num]
users = part_data['reviewerID'].drop_duplicates().values
user_dict = {u: idx for idx, u in enumerate(users)}
items = part_data['asin'].drop_duplicates().values
item_dict = {i: idx for idx, i in enumerate(items)}
print(len(user_dict), len(item_dict))
ui_matrix = np.mat([len(user_dict), len(item_dict)])
part_data = shuffle(part_data)
part_data['uid'] = part_data['reviewerID'].apply(lambda uid: user_dict.get(uid))
part_data['mid'] = part_data['asin'].apply(lambda tid: item_dict.get(tid))
part_data = part_data[['uid', 'mid', 'overall']]

# part_data[['uid', 'mid']] = part_data[['uid', 'mid']]
train_data, test_data = part_data[0: int(record_num*0.8)], \
                                  part_data[int(record_num*0.8):]
print(test_data[:10])
svd = SVD(train_data.values)
svd.train()





