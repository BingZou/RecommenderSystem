from time import time
from random import sample 
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.ml.feature import StringIndexer
import os

'''
To Run:
spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g Indexer.py
'''


def Indexer(spark,train_address, val_address, tst_address):

    beg = time()
    df_train = spark.read.parquet(train_address)
    df_val = spark.read.parquet(val_address)
    df_test = spark.read.parquet(tst_address)

    print('File Reading Finished')

    # subsample 
    # here we include the last 110K users (for validation/testing)
    # want to include all 110K users at the end and randomly draw 10% others
    subsample_frac = 0.1
    all_user_ids =  [row['user_id'] for row in df_train.select('user_id').distinct().collect()]
    val_user_ids = [row['user_id'] for row in df_val.select('user_id').distinct().collect()]
    test_user_ids = [row['user_id'] for row in df_test.select('user_id').distinct().collect()]
    train_user_ids = list(set(all_user_ids) - set(val_user_ids) - set(test_user_ids))
    selected_train_ids = sample(train_user_ids,round(len(train_user_ids) * 0.2))

    # >>> len(all_user_ids)
    # 1129318
    # >>> len(val_user_ids)
    # 10000
    # >>> len(test_user_ids)
    # 100000
    # >>> len(train_user_ids)
    # 1019318

    df_train = df_train.where(df_train.user_id.isin(selected_train_ids + val_user_ids + test_user_ids))

    print('Sampling Finished')

    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_numeric")
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_numeric")
    
    model = Pipeline(stages=[user_indexer, track_indexer]).fit(df_train.union(df_val).union(df_test))
    df_train,df_val,df_test = [model.transform(x) for x in (df_train,df_val,df_test)]


    df_train = df_train.select("user_id_numeric","track_id_numeric","count")
    df_val = df_val.select("user_id_numeric","track_id_numeric","count")
    df_test = df_test.select("user_id_numeric","track_id_numeric","count")

    print('Formatting Finished')

    df_train_subsampled.write.parquet("./train_formatted.parquet", mode='overwrite')
    df_val.write.parquet("./val_formatted.parquet", mode='overwrite')
    df_test.write.parquet("./test_formatted.parquet", mode='overwrite')
    end = time()
    print('Indexer and Subsampler succeed. Took %f s' %(end-beg))
    return 
if __name__ == "__main__":

    beg = time()
    spark = SparkSession.builder.appName('main').getOrCreate()

    train_address = "hdfs:/user/bm106/pub/project/cf_train.parquet"
    val_address = "hdfs:/user/bm106/pub/project/cf_validation.parquet"
    tst_address = "hdfs:/user/bm106/pub/project/cf_test.parquet"
    Indexer(spark,train_address, val_address, tst_address)

    # increase HDFS replication Factor:
    os.system('hadoop fs -setrep -w 10 ./train_formatted.parquet')
    os.system('hadoop fs -setrep -w 10 ./val_formatted.parquet')
    os.system('hadoop fs -setrep -w 10 ./test_formatted.parquet')

    # Copy parquet to local env:
    os.system('hadoop fs -get ./train_formatted.parquet')
    os.system('hadoop fs -get ./val_formatted.parquet')
    os.system('hadoop fs -get ./test_formatted.parquet')

