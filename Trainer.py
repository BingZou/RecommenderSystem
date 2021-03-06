from time import time
from pyspark.sql import SparkSession
from pyspark.ml.recommendation import ALS
import os
'''
To Run:
spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g Trainer.py
'''

def Trainer(spark,df_train,rank,regParam,alpha,K = 500):
    output_file = 'ALSModel_%s_%s__%s' %(str(rank),str(regParam),str(alpha))
    FileExistFlag = os.system('hadoop fs -test -e %s' %output_file)
    if not FileExistFlag == 0:
        beg = time()
        als = ALS(rank=rank, maxIter=10, regParam=regParam, alpha=alpha, implicitPrefs = True,
              userCol="user_id_numeric", itemCol="track_id_numeric", ratingCol="count",
                    coldStartStrategy="drop")
        model = als.fit(df_train)
        print('Train Finished')
        model.write().overwrite().save(output_file)
        end = time()
        print('ALSModel_%s_%s__%s Saved. Took %f s' %(str(rank),str(regParam),str(alpha),end-beg))
    else:
        print('ALSModel_%s_%s__%s Already Exist.' %(str(rank),str(regParam),str(alpha)))
    return

if __name__ == "__main__":
    spark = SparkSession.builder.appName('main').getOrCreate()

    formatted_train_address = "data/train_formatted.parquet"
    df_train = spark.read.parquet(formatted_train_address)
    print('File Reading Finished')
    for rank in [5,10,15]:
        for regParam in [0.01,0.1,0.2,0.3,0.5]:
            for alpha in [0.5,1,3,5,10]:
                Trainer(spark,df_train,rank,regParam,alpha)