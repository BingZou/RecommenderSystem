module load spark/2.4.0
module load python/gnu/3.6.5
cd ./final_project

alias spark-submit='PYSPARK_PYTHON=$(which python) spark-submit'
#spark-submit --conf spark.driver.memory=8g --conf spark.executor.memory=8g Indexer.py
nohup spark-submit --conf spark.driver.memory=32g --conf spark.executor.memory=32g Trainer.py
spark-submit --conf spark.driver.memory=16g --conf spark.executor.memory=16g Tester.py