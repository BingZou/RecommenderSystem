from time import time
import pandas as pd
from pyspark.sql import Window
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.ml.recommendation import ALSModel
from pyspark.mllib.evaluation import RankingMetrics

'''
To Run:
spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g Tester.py
'''

def Tester(spark,df_test,rank,regParam,alpha, K = 500):
    beg = time()
    model = ALSModel.load('ALSModel_%s_%s__%s' %(str(rank),str(regParam),str(alpha)))
    print('ALSModel_%s_%s__%s Loaded' %(str(rank),str(regParam),str(alpha)))

    targetUsers = df_test.select("user_id_numeric").distinct()
    userRecs = model.recommendForUserSubset(targetUsers,K)
    userRecs = userRecs.select("user_id_numeric", "recommendations.track_id_numeric", "recommendations.rating")


    # need to get ordered list of track_id based on counts groupby individual users.
    # reference:https://stackoverflow.com/questions/46580253/collect-list-by-preserving-order-based-on-another-variable
    w = Window.partitionBy("user_id_numeric").orderBy(df_test['count'].desc())
    labels = df_test.withColumn('ActualRanking', F.collect_list("track_id_numeric").over(w))
    labels = labels.select(['user_id_numeric','ActualRanking']).dropDuplicates(['user_id_numeric'])

    # Get the metrics
    # predictionsAndlabels should be an RDD of (predicted ranking, ground truth set) pairs.
    # reference: https://spark.apache.org/docs/2.2.0/api/python/pyspark.mllib.html#pyspark.mllib.evaluation.RankingMetrics
    predictionsAndlabels = userRecs.join(labels, [labels.user_id_numeric==userRecs.user_id_numeric],'left').select('track_id_numeric', 'ActualRanking')
    print('Getting Metrics')
    metricsRank = RankingMetrics(predictionsAndlabels.rdd)
    p10 = round(metricsRank.precisionAt(10),5)
    p500 = round(metricsRank.precisionAt(500),5)
    map_ = round(metricsRank.meanAveragePrecision,5)
    ndcg = round(metricsRank.ndcgAt(500),5)
    end = time()
    print ("------------------------------------------")
    print ("Params: Rank %f | regParam %f | alpha = %f" % (rank, regParam, alpha))
    print ("p(10)   %.8f" % p10)
    print ("p(500)   %.8f" % p500)
    print ("MAP  %.8f" % map_)
    print ("nDCG %.8f" % ndcg)
    print ("Took %f s" %(end-beg))

    with open("validatorLog.txt", "a") as log:
        log.write(','.join([str(rank),str(regParam),str(alpha),str(p10),str(p500),str(map_),str(ndcg),str(end-beg)]) + '\n')

    return [rank,regParam,alpha,p10,p500,map_,ndcg,(end-beg)]

if __name__ == "__main__":

    spark = SparkSession.builder.appName('main').getOrCreate()

    #validation step
    formatted_test_address = "data/val_formatted.parquet"
    df_test = spark.read.parquet(formatted_test_address)

    tested_combinations = pd.read_csv("validatorLog.txt",header = None)[[0,1,2]].values.tolist()

    results = []
    # for rank in [10,15]:
    #     for regParam in [0.01,0.1,0.2,0.5]:
    #         for alpha in [0.5,1,3,5,10]:
    for rank in [50,100,150]:
        for regParam in [0.5]:
            for alpha in [1,10,50]:
                params  = [rank,regParam,alpha]
    # params  = [120,40,0.2]
    if params not in tested_combinations:
        results.append(Tester(spark,df_test,rank,regParam,alpha, K = 500))
    else:
        pass


    df = pd.DataFrame(results)
    df.to_csv('Validation_results.csv',index = False)


    # #test step
    # rank,regParam,alpha = [5,0.01,0.5] #optimal params to be input
    # formatted_test_address = "./data/test_formatted.parquet"
    # df_test = spark.read.parquet(formatted_test_address)
    # Tester(spark,df_test,rank,regParam,alpha, K = 500)