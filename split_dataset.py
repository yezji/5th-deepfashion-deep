import pandas as pd
import os
import shutil
from sklearn.model_selection import train_test_split


def split_dataset():
    df = pd.read_csv('./dataset/c_dataset/labels_v2.1.csv')  # 7:1.5:1.5
    X_train, X_test = train_test_split(df, test_size=.3, random_state=1234, shuffle=True)
    # X_valid, X_test = train_test_split(X_test, test_size=.5, random_state=1234, shuffle=True)

    X_train.to_csv("./dataset/c_dataset/train.csv", mode='w', header=True, index=False)
    # X_valid.to_csv("./dataset/c_dataset/val.csv", mode='w', header=True, index=False)
    X_test.to_csv("./dataset/c_dataset/test.csv", mode='w', header=True, index=False)

    return 0


def move_images():
    res = os.listdir('./dataset/c_dataset/from_images/')
    # mode = ['train', 'val', 'test']
    mode = ['train', 'test']
    tar = './dataset/c_dataset/images/'

    for folder in mode:
        df = pd.read_csv('dataset/c_dataset/' + folder + '.csv', sep=',')
        dir = tar + folder + '/'
        if os.path.exists(tar + folder) is False:
            os.mkdir(tar + folder)

        col_df = df['image_file_name']
        for i in col_df:
            category = i.split('_0')[0]
            if os.path.exists(dir + category) is False:
                os.mkdir(dir + category)

            for part in res:
                if category in part:
                    res_path = './dataset/c_dataset/from_images/' + part + '/'
                    for num in os.listdir(res_path):
                        if num in i:
                            shutil.move((res_path + num), dir + category + '/' + num)
                            print(folder, res_path+num, '>>>>>>>>>>>>', dir + category + '/' + num)
    return 0


print(split_dataset())
print(move_images())