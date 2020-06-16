module load spark/2.4.0
module load python/gnu/3.6.5
cd ./final_project


pyspark


#First of all some EDA

#read table
df_train = spark.read.parquet("hdfs:/user/bm106/pub/project/cf_train.parquet")
df_val = spark.read.parquet("hdfs:/user/bm106/pub/project/cf_validation.parquet")
df_test = spark.read.parquet("hdfs:/user/bm106/pub/project/cf_test.parquet")

>>> df_train.describe().show()
+-------+--------------------+-----------------+------------------+--------------------+
|summary|             user_id|            count|          track_id|   __index_level_0__|
+-------+--------------------+-----------------+------------------+--------------------+
|  count|            48373586|         48373586|          48373586|            48373586|
|   mean|                null|2.866858847305635|              null|        2.41867925E7|
| stddev|                null|6.437724686877204|              null|1.3964251593721591E7|
|    min|00000b72200188206...|                1|TRAAAAK128F9318786|                   0|
|    max|fffff9534445f481b...|             9667|TRZZZZZ12903D05E3A|            48373585|
+-------+--------------------+-----------------+------------------+--------------------+

>>> df_val.describe().show()
+-------+--------------------+------------------+-----------------+-----------------+
|summary|             user_id|           song_id|            count|__index_level_0__|
+-------+--------------------+------------------+-----------------+-----------------+
|  count|              131039|            131039|           131039|           131039|
|   mean|                null|              null|3.168659711994139|          65519.0|
| stddev|                null|              null|7.087232233049278|37827.84529946161|
|    min|0007140a3796e901f...|SOAAAGQ12A8C1420C8|                1|                0|
|    max|fffa8a20b865c4d24...|SOZZZWN12AF72A1E29|              646|           131038|
+-------+--------------------+------------------+-----------------+-----------------+

>>> df_test.describe().show()
+-------+--------------------+------------------+------------------+-----------------+
|summary|             user_id|           song_id|             count|__index_level_0__|
+-------+--------------------+------------------+------------------+-----------------+
|  count|              135938|            135938|            135938|           135938|
|   mean|                null|              null|3.1634568700436962|          67968.5|
| stddev|                null|              null| 7.036054663228981|39242.06478385152|
|    min|0007140a3796e901f...|SOAAADZ12A8C1334FB|                 1|                0|
|    max|fffa8a20b865c4d24...|SOZZZPV12A8C1444B5|               787|           135937|
+-------+--------------------+------------------+------------------+-----------------+

# find all distinct users by user_id 

all_user_ids =  [row['user_id'] for row in df_train.select('user_id').distinct().collect()]
val_user_ids = [row['user_id'] for row in df_val.select('user_id').distinct().collect()]
test_user_ids = [row['user_id'] for row in df_test.select('user_id').distinct().collect()]

>>> len(all_user_ids)
1129318
>>> len(val_user_ids)
10000
>>> len(test_user_ids)
100000
>>> val_user_ids == test_user_ids
False

# do subsample for easier training and testing
train_user_ids = list(set(all_user_ids) - set(val_user_ids) - set(test_user_ids))
>>> len(train_user_ids)
1019318

# to fastly build and test model, we subsample 10% of the original data
from random import sample 
selected_train_ids = sample(train_user_ids,10**5)
selected_val_ids = sample(val_user_ids,10**3)
selected_test_ids = sample(test_user_ids,10**4)

df_train_subsampled = df_train.where(df_train.user_id.isin(selected_train_ids + 
															selected_val_ids + selected_test_ids))
df_val_subsampled = df_val.where(df_val.user_id.isin(selected_val_ids))
df_test_subsampled = df_test.where(df_test.user_id.isin(selected_test_ids))


>>> df_train_subsampled.describe().show()
+-------+--------------------+-----------------+------------------+--------------------+
|summary|             user_id|            count|          track_id|   __index_level_0__|
+-------+--------------------+-----------------+------------------+--------------------+
|  count|             4885254|          4885254|           4885254|             4885254|
|   mean|                null|2.878344094288649|              null|2.3508064704627436E7|
| stddev|                null| 6.13360479183715|              null|1.4302100698948955E7|
|    min|0000267bde1b3a70e...|                1|TRAAAAK128F9318786|                   0|
|    max|ffffd330940a2a407...|             1222|TRZZZZZ12903D05E3A|            48373198|
+-------+--------------------+-----------------+------------------+--------------------+

>>> df_val_subsampled.describe().show()
+-------+--------------------+-----------------+------------------+-----------------+
|summary|             user_id|            count|          track_id|__index_level_0__|
+-------+--------------------+-----------------+------------------+-----------------+
|  count|               13779|            13779|             13779|            13779|
|   mean|                null|3.165686914870455|              null|70284.87451919587|
| stddev|                null|9.464055724695942|              null| 38364.3510402961|
|    min|00363991358e0f5ab...|                1|TRAACPH12903CF5F14|               59|
|    max|ffdfc7f9864ee172c...|              787|TRZZZYR128F92F0796|           135888|
+-------+--------------------+-----------------+------------------+-----------------+

>>> df_test_subsampled.describe().show()
+-------+--------------------+------------------+------------------+-----------------+
|summary|             user_id|             count|          track_id|__index_level_0__|
+-------+--------------------+------------------+------------------+-----------------+
|  count|              137191|            137191|            137191|           137191|
|   mean|                null|3.1476627475563266|              null|680924.9059632191|
| stddev|                null| 7.522489375415325|              null|390215.0161565848|
|    min|00015189668691680...|                 1|TRAAAED128E0783FAB|               21|
|    max|fffe5b73c50c72ca9...|              1267|TRZZZYX128F92D32C6|          1368402|
+-------+--------------------+------------------+------------------+-----------------+

all_user_ids_subsampled =  [row['user_id'] for row in df_train_subsampled.select('user_id').distinct().collect()]
val_user_ids_subsampled = [row['user_id'] for row in df_val_subsampled.select('user_id').distinct().collect()]
test_user_ids_subsampled = [row['user_id'] for row in df_test_subsampled.select('user_id').distinct().collect()]

assert(all_user_ids_subsampled == 111000)
assert(val_user_ids_subsampled == 1000)
assert(val_user_ids_subsampled == 10000)

df_train_subsampled.write.parquet("./cf_train_subsampled.parquet")
df_val_subsampled.write.parquet("./cf_validation_subsampled.parquet")
df_test_subsampled.write.parquet("./cf_test_subsampled.parquet")


#buildmodel
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.sql import Row
from pyspark.ml.feature import StringIndexer

df_train = spark.read.parquet("./cf_train_subsampled.parquet")
df_val = spark.read.parquet("./cf_validation_subsampled.parquet")
df_test = spark.read.parquet("./cf_test_subsampled.parquet")

df_train.createOrReplaceTempView('df_train')
df_val.createOrReplaceTempView('df_val')
df_test.createOrReplaceTempView('df_test')

df_train = StringIndexer(inputCol="user_id", outputCol="user_id_numeric").fit(df_train).transform(df_train)
df_train = StringIndexer(inputCol="track_id", outputCol="track_id_numeric").fit(df_train).transform(df_train)
df_train.show()
# Build the recommendation model using ALS on the training data
# Note we set cold start strategy to 'drop' to ensure we don't get NaN evaluation metrics
als = ALS(rank=10, maxIter=10, regParam=0.1, alpha=1.0, 
      userCol="user_id_numeric", itemCol="track_id_numeric", ratingCol="count",
            coldStartStrategy="drop")
model = als.fit(df_train)

# Evaluate the model by computing the RMSE on the test data
predictions = model.transform(test)
evaluator = RegressionEvaluator(metricName="rmse", labelCol="rating",
                                predictionCol="prediction")
rmse = evaluator.evaluate(predictions)
print("Root-mean-square error = " + str(rmse))

# Generate top 10 movie recommendations for each user
userRecs = model.recommendForAllUsers(10)
# Generate top 10 user recommendations for each movie
movieRecs = model.recommendForAllItems(10)

from pyspark.ml import Pipeline
from pyspark.ml.tuning import CrossValidator, ParamGridBuilder
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.recommendation import ALS
from pyspark.ml.feature import StringIndexer

data_file = "./cf_train_subsampled.parquet"

df = spark.read.parquet(data_file)
UserIDscaler = StringIndexer(inputCol="user_id", outputCol="user_id_numeric")
TrackIDscaler = StringIndexer(inputCol="track_id", outputCol="track_id_numeric")

als = ALS(rank=10, maxIter=10, regParam=0.1, alpha=1.0, 
      userCol="user_id_numeric", itemCol="track_id_numeric", ratingCol="count",
            coldStartStrategy="drop",implicitPrefs=True)

paramGrid = (ParamGridBuilder()
             .addGrid(als.rank, range(1,11))
             .addGrid(lr.regParam, [x/10 for x in range(1,11)])
             .addGrid(lr.alpha, [x/10 for x in range(1,11)])
             .build())

pipeline = Pipeline(stages=[UserIDscaler,TrackIDscaler,als])
crossval = CrossValidator(estimator=pipeline,
                          estimatorParamMaps=paramGrid,
                          evaluator=MulticlassClassificationEvaluator(),
                          numFolds=5) 

cvModel = crossval.fit(df)