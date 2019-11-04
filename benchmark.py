# coding: utf-8
import os
import multiprocessing as mp
import pandas as pd
import numpy as np

from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error

from funk_svd.dataset import fetch_ml20m_ratings
from funk_svd.utils import timer
from funk_svd import SVD

import surprise  #Scikit-Learn library for recommender systems. 

import logging
# https://climate-cms.org/2018/10/05/introduction-to-python-logging.html
logger = logging.getLogger('mf')
logging.basicConfig(level=logging.DEBUG, format="%(asctime)s\t%(message)s")

def data0():
    # [MovieLens 20M Dataset Research Paper]("http://files.grouplens.org/papers/harper-tiis2015.pdf")
    # get_ipython().run_cell_magic('time', '', '\ndf = fetch_ml20m_ratings()\nprint()')
    logger.info('Before load MovieLens dataset')
    df = fetch_ml20m_ratings()
    # columns = [u_id, i_id, rating, timestamp]
    df.head()
    df.tail()
    logger.info('After load MovieLens dataset')
    logger.info('After load raw.shape=', raw.shape)
    logger.info('We have', raw.shape[0], 'ratings')
    return df

def data1():
    raw = pd.read_parquet('/data/user/jovyan/ppmi.parquet')
    logger.info('After load raw.shape=%r', raw.shape)
    raw.drop_duplicates(inplace=True)
    # columns = [u_id, i_id, rating, timestamp]
    raw['u_id'] = raw['u_id'].astype(np.int32)
    raw['i_id'] = raw['i_id'].astype(np.int32)
    logger.info('We have %d ratings', raw.shape[0])
    logger.info('The number of unique users we have is: %d ', len(raw.user_id.unique()))
    logger.info('The number of unique books we have is: %d ', len(raw.book_id.unique()))
    logger.info('The median user rated %d books.' % raw.user_id.value_counts().median())
    logger.info('The max rating is: %d' % raw.rating.max())
    logger.info('the min rating is: %d' % raw.rating.min())
    raw.head()
    raw.tail()
    # Return the shifted positive PMI.
    df1 = pd.DataFrame({ 'u_id': raw['user_id'], 'i_id': raw['book_id'], 'rating': raw['rating'].clip(lower=0.0)})
    return df1


# ## Perform a train/val/test split

# There is 138,493 different users in the MovieLens20m dataset, each of them having rated at least 20 movies.
# Let's sample the 4 last ratings per user and randomly split them between validation and test sets. 
# 
# To do so, we need to query our DataFrame for each user and then select their 4 last ratings.
# With so much users it's naturally quite expensive... hopefully it's possible to parallelize it as iterations are independant,
# allowing us to save some time (especially if you have good computing ressources).
# I'm using an Intel Core i5-7300U CPU (only 2 physical cores) on a 16GB laptop so I won't be able to save that much :)
# 
# <img src="https://www.dlapiper.com/~/media/images/insights/publications/2015/warning.jpg?la=en&hash=6F2E30889FD9E0B11016A1712E6E583575717C54" width="23" align="left">
# 
# If you want to run this notebook with **Windows**, you won't be able to use `multiprocessing.Pool` because it's lacking `fork()`.
# For simplicity you can also do it sequentially without loosing so much time compared to my dual core CPU.


@timer(text="")
def compute_val_test_mask(users, df, i, n_process, n_rate=4):
    val_test_mask = []
    for j in range(i, len(users), n_process):
        u_id = users[j]
        u_subset = df[df["u_id"] == u_id].copy()
        val_test_mask += u_subset.iloc[-n_rate:].index.tolist()
    return val_test_mask

def runit(df, edim=15):
    """Example usage: df1 = data1(); runit(df1.iloc[:1000000], edim=15)"""
    logger.info('df.shape=%r', df.shape)
    users = df["u_id"].unique()
    if True:
        # This one CPU version takes 45 seconds for wombat at 122 million obs.
        logger.info("Before sample for training set")
        train = df.sample(frac=0.8, random_state=7)
        logger.info("Before sample for xval set")
        xval = df.drop(train.index.tolist()).sample(frac=0.5, random_state=8)
        logger.info("Before sample for test set")
        test = df.drop(train.index.tolist()).drop(xval.index.tolist())
        logger.info("After [train, xval, test] sampling")
    else:
        # This currently hangs in multiprocessing
        logger.info("Before compute val test mask")
        n_process = os.cpu_count() 
        n_process = 12
        pool = mp.Pool(processes=n_process)
        results = [
            pool.apply_async(compute_val_test_mask,
            args=(users, df, i, n_process))
            for i in range(n_process)
        ]
        logger.info("Got results for compute xval test mask")
        results = [p.get() for p in results]
        val_test_mask = [item for sublist in results for item in sublist]
        logger.info("After compute xval test mask")
        train = df.drop(val_test_mask)
        xval = df.loc[val_test_mask].sample(frac=0.5, random_state=7)
        test = df.loc[val_test_mask].drop(xval.index.tolist())

    # ## Modelization
    # Let's fit our model.
    logger.info("Before compute Funk SVD")
    svd = SVD(learning_rate=0.001, regularization=0.005, n_epochs=100,
              n_factors=edim, min_rating=0.0, max_rating=20.0)
    logger.info("Before fit Funk SVD")
    svd.fit(X=train, X_val=xval, early_stopping=True, shuffle=False)
    logger.info("After fit Funk SVD")
    logger.info("Predict for test set and compare with reality. Root mean squared error and mean absolute error.")
    pred = svd.predict(test)
    rmse = np.sqrt(mean_squared_error(test["rating"], pred))
    mae = mean_absolute_error(test["rating"], pred)
    logger.info(f"  Test RMSE: {rmse:.2f}")
    logger.info(f"  Test  MAE:  {mae:.2f}")
    logger.info(f"Save embedding vectors here.")
    return svd



def comparison_with_surprise(edim=15):
    # ## Comparison with Surprise library
    from surprise import Dataset
    from surprise import Reader
    from surprise import SVD
    # Format data according Surprise way.
    #get_ipython().run_cell_magic('time', '', '''
    #        \nreader = Reader(rating_scale=(1, 5))\n
    #        \ntrainset = Dataset.load_from_df(train[["u_id", "i_id", "rating"]],
    #        \nreader=reader).build_full_trainset()\n
    #        \ntestset = Dataset.load_from_df(test[["u_id", "i_id", "rating"]], reader=reader)
    #        \ntestset = testset.construct_testset(testset.raw_ratings)''')
    reader = Reader(rating_scale=(1, 5))
    trainset = Dataset.load_from_df(train[["u_id", "i_id", "rating"]], reader=reader).build_full_trainset()
    testset = Dataset.load_from_df(test[["u_id", "i_id", "rating"]], reader=reader)
    testset = testset.construct_testset(testset.raw_ratings)
    # Fit the model with the same parameters.
    svd = SVD(lr_all=0.001, reg_all=0.005, n_epochs=46, n_factors=edim, verbose=True)
    svd.fit(trainset)
    print()
    # Predict test set and compute results.
    pred = svd.test(testset)
    y_true = [p.r_ui for p in pred]
    y_hat = [p.est for p in pred]
    rmse = np.sqrt(mean_squared_error(y_true, y_hat))
    mae = mean_absolute_error(y_true, y_hat)
    print("Test RMSE: {:.2f}".format(rmse))
    print("Test MAE:  {:.2f}".format(mae))
    print()
    # Accuracy performance is naturally equivalent, difference stands in the computation time, `Numba` allowing us to run more than 10 times faster than with cython.
    # 
    # | Movielens 20M | RMSE   | MAE    | Time          |
    # |:--------------|:------:|:------:|--------------:|
    # | Surprise      |  0.88  |  0.68  | 11 min 13 sec |
    # | Funk-svd      |  0.88  |  0.68  |        48 sec |
