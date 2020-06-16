from time import time
from pyspark.sql import Window
from pyspark.ml import Pipeline
from pyspark.sql import SparkSession
from pyspark.conf import SparkConf
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer
from pyspark.mllib.evaluation import RankingMetrics



def Indexer(spark,train_address, val_address, tst_address, repartition_size = 10000):

    df_train = spark.read.parquet(train_address)
    df_val = spark.read.parquet(val_address)
    df_test = spark.read.parquet(tst_address)

    # user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_numeric").fit(df_train)
    # track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_numeric").fit(df_train.union(df_val))

    # df_train = user_indexer.transform(df_train)
    # df_train = track_indexer.transform(df_train)
    # df_val = user_indexer.transform(df_val)
    # df_val = track_indexer.transform(df_val)
    # df_train = df_train.select("user_id_numeric","track_id_numeric","count")
    # df_val = df_val.select("user_id_numeric","track_id_numeric","count")
    # df_test = user_indexer.transform(df_test)
    # df_test = track_indexer.transform(df_test)


    user_indexer = StringIndexer(inputCol="user_id", outputCol="user_id_numeric")
    track_indexer = StringIndexer(inputCol="track_id", outputCol="track_id_numeric")
    
    # fit df_train only
    model = Pipeline(stages=[user_indexer, track_indexer]).fit(df_train)
    df_train,df_val,df_test = [model.transform(x) for x in (df_train,df_val,df_test)]

    df_train = df_train.select("user_id_numeric","track_id_numeric","count")
    df_val = df_val.select("user_id_numeric","track_id_numeric","count")
    df_test = df_test.select("user_id_numeric","track_id_numeric","count")
    
    # df_train = df_train.repartition(repartition_size,"user_id_numeric","track_id_numeric")
    # df_val = df_val.repartition(repartition_size,"user_id_numeric","track_id_numeric")
    # df_test = df_test.repartition(repartition_size,"user_id_numeric","track_id_numeric")

    # df_train.write.parquet("./train_formatted.parquet", mode='overwrite')
    # df_val.write.parquet("./val_formatted.parquet", mode='overwrite')
    # df_test.write.parquet("./test_formatted.parquet", mode='overwrite')

    print('Indexer succeed.')
    return df_train,df_val,df_test


def Trainer(spark,df_train,rank,regParam,alpha,K = 500):
    print('Trainer Start.')
    #df_train = spark.read.parquet(formatted_train_address)
    #df_train = df_train.withColumn('count', F.log('count')) #takes log

    als = ALS(rank=rank, maxIter=10, regParam=regParam, alpha=alpha, implicitPrefs = True,
          userCol="user_id_numeric", itemCol="track_id_numeric", ratingCol="count",
                coldStartStrategy="drop")
    model = als.fit(df_train)

    print('ALS succeed.')
    return model


def Tester(spark,model,df_test,rank,regParam,alpha, K = 500):
    #df_test = spark.read.parquet(formatted_test_address)
    targetUsers = df_test.select("user_id_numeric").distinct()
    userRecs = model.recommendForUserSubset(targetUsers,K)
    userRecs = userRecs.select("user_id_numeric", "recommendations.track_id_numeric", "recommendations.rating")


    # need to get ordered list of track_id based on counts groupby individual users.
    # reference:https://stackoverflow.com/questions/46580253/collect-list-by-preserving-order-based-on-another-variable
    w = Window.partitionBy("user_id_numeric").orderBy(df_val['count'].desc())
    labels = df_val.withColumn('ActualRanking', F.collect_list("track_id_numeric").over(w))
    labels = labels.select(['user_id_numeric','ActualRanking']).dropDuplicates(['user_id_numeric'])

    # Get the metrics
    # predictionsAndlabels should be an RDD of (predicted ranking, ground truth set) pairs.
    # reference: https://spark.apache.org/docs/2.2.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics
    predictionsAndlabels = userRecs.join(labels, [labels.user_id_numeric==userRecs.user_id_numeric],'left').select('track_id_numeric', 'ActualRanking')
    metricsRank = RankingMetrics(predictionsAndlabels.rdd)

    print ("------------------------------------------")
    print ("Params: Rank %f | regParam %f | alpha = %f" % (rank, regParam, alpha))
    print ("p(15)   %.8f" % metricsRank.precisionAt(15))
    print ("p(500)   %.8f" % metricsRank.precisionAt(500))
    print ("MAP  %.8f" % metricsRank.meanAveragePrecision)
    print ("nDCG %.8f" % metricsRank.ndcgAt(K))
    return

if __name__ == "__main__":

    beg = time()
    conf = SparkConf()
    conf.set("spark.executor.memory", "32G")
    conf.set("spark.driver.memory", '32G')
    conf.set("spark.executor.cores", "4")
    conf.set('spark.executor.instances', '32')
    conf.set("spark.serializer", "org.apache.spark.serializer.KryoSerializer")
    conf.set("spark.shuffle.service.enabled", "false")
    conf.set("spark.dynamicAllocation.enabled", "false")
    conf.set('spark.sql.broadcastTimeout', '36000')
    conf.set("spark.default.parallelism", "40")
    conf.set("spark.sql.shuffle.partitions", "40")
    conf.set("spark.io.compression.codec", "snappy")
    conf.set("spark.rdd.compress", "true")
    
    spark = SparkSession.builder.appName('main').getOrCreate()

    train_address = "hdfs:/user/bm106/pub/project/cf_train.parquet"
    val_address = "hdfs:/user/bm106/pub/project/cf_validation.parquet"
    tst_address = "hdfs:/user/bm106/pub/project/cf_test.parquet"

    # train_address = "./cf_train_subsampled.parquet"
    # val_address = "./cf_validation_subsampled.parquet"
    # tst_address = "./cf_test_subsampled.parquet"


    for params in [(1,0.01,5)]:
        # rank = 10
        # regParam = 0.1
        # alpha = 5
        rank,regParam,alpha = params
        df_train,df_val,df_test = Indexer(spark,train_address, val_address, tst_address)
        t1 = time()
        print(t1-beg)
        model = Trainer(spark,df_train,rank,regParam,alpha,K = 500)
        t2 = time()
        print(t2-t1)
        Tester(spark,model,df_val,rank,regParam,alpha, K = 500)
        t3 = time()
        print(t3-t2)
