import pandas as pd
import numpy as np
from sklearn.utils import shuffle
from svd_baseline import SVD
import csv


def process_core_5(ui_matrix, core=5):
    print(ui_matrix.count())
    u_matrix = ui_matrix[['reviewerID', 'overall']].copy().groupby('reviewerID').size().reset_index(name='counts')
    u_matrix = u_matrix.loc[u_matrix['counts'] >= core]
    i_matrix = ui_matrix[['asin', 'overall']].copy().groupby('asin').size().reset_index(name='counts')
    i_matrix = i_matrix.loc[i_matrix['counts'] >= core]
    u_list = u_matrix['reviewerID'].values
    i_list = i_matrix['asin'].values
    print(len(u_list))
    print(len(i_list))
    ui_matrix = ui_matrix.loc[ui_matrix['reviewerID'].isin(u_list)]
    ui_matrix = ui_matrix.loc[ui_matrix['asin'].isin(i_list)]
    print(ui_matrix.count())
    return ui_matrix


ui_matrix = pd.read_csv('./data/fashion/UIMatrix.csv')
ui_matrix = shuffle(ui_matrix)
record_num = len(ui_matrix)
train_data, test_data = ui_matrix[0: int(record_num * 0.8)], \
                        ui_matrix[int(record_num * 0.8):]
ave = np.array(train_data.values[:, 2]).mean()
# train_data = process_core_5(train_data, core=3)
svd = SVD(train_data.values, ave=ave, k=20)
for i in range(1):
    print('round %s' % i)
    svd.train(steps=2, Lambda=1.2)
    total_rmse = 0
    for line in test_data.values:
        user, item, score = line
        r = svd.pred(user, item)
        total_rmse += (r - score) ** 2
    print('rmse: ', np.sqrt(total_rmse / len(test_data)))
    print()

records = []
user_codes = pd.read_csv('./data/fashion/user_code.csv')
item_codes = pd.read_csv('./data/fashion/item_code.csv')
with open('./data/AmazonFashionWithImgPartitioned_rating_test_unlabel.csv') as path:
    file = csv.reader(path)
    for user_id, item_id in file:
        user_code, item_code = user_codes.loc[int(user_id)][1], item_codes.loc[int(item_id)][1]
        score = svd.pred(user_code, item_code)
        record = [user_id, item_id, score]
        records.append(record)
with open('./data/fashion/rating_fashion.csv', 'w', newline='\n') as path:
    file = csv.writer(path)
    file.writerows(records)







