from multiprocessing.pool import ThreadPool
import time


def get_rs(args):

    from pyspark.ml.evaluation import RegressionEvaluator
    from pyspark.ml.recommendation import ALS
    from pyspark.ml.feature import StringIndexer
    from pyspark.sql import SparkSession
    import random
    import string

    def id_generator(size=6, chars=string.ascii_uppercase + string.digits):
       return ''.join(random.choice(chars) for _ in range(size))

    spark = SparkSession.builder.appName('Session_%s' %id_generator()).getOrCreate()
    df_train = spark.read.parquet("./cf_train_subsampled.parquet")
    df_val = spark.read.parquet("./cf_validation_subsampled.parquet")
    df_test = spark.read.parquet("./cf_test_subsampled.parquet")

    # train contains all user, but not all tracks
    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_numeric").fit(df_train)
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_numeric").fit(df_train.union(df_val))

    df_train = user_indexer.transform(df_train)
    df_train = track_indexer.transform(df_train)
    df_val = user_indexer.transform(df_val)
    df_val = track_indexer.transform(df_val)
    df_test = user_indexer.transform(df_test)
    df_test = track_indexer.transform(df_test)

    rank,regParam,alpha = args
    als = ALS(rank=rank, maxIter=10, regParam=regParam, alpha=alpha, implicitPrefs = True,
          userCol="user_id_numeric", itemCol="track_id_numeric", ratingCol="count",
                coldStartStrategy="drop")
    #model = als.trainImplicit(df_train)
    model = als.fit(df_train)

    # Evaluate the model by computing the RMSE on the test data
    predictions = model.transform(df_val)
    evaluator = RegressionEvaluator(metricName="rmse", labelCol="count",
                                    predictionCol="prediction")
    rmse = evaluator.evaluate(predictions)
    return [rank,regParam,alpha,rmse]

if __name__ == '__main__':
    start = time.time()

    params = []
    for rank in [1,5,10,15,20,25,30]:
      for regParam in [0.01,0.1,1,10]:
          for alpha in [1,2,3,4,5,10,15,20]:
                params += [(rank,regParam,alpha)]

    p = ThreadPool(4)
    results = p.map(get_rs,params)

    for line in results:
        rank,regParam,alpha,rmse = line
        print ('Rank = %s, regParam = %s, alpha = %s: RMSE = %0.5f' %(str(rank),str(regParam),str(alpha),rmse))

    end = time.time()
    print(end - start)