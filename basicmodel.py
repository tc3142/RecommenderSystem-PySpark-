

# -*- coding: utf-8 -*-


#importing essential libraries
import sys
import time
import getpass
import itertools
from pyspark.sql import SparkSession
import pyspark.sql.functions as F
from pyspark.sql.functions import *
from pyspark.sql.window import Window
def main(spark,file_path1, file_path2, file_path3):
  seed=200000
  #reading parquet files
  train_df = spark.read.parquet(file_path1)
  val_df = spark.read.parquet(file_path2)
  test_df = spark.read.parquet(file_path3)
  #for sampling the dataframes
  #train_df= train_1.sample(0.01, 200000)
  #val_df= val_1.sample(0.01,200000)
  #test_df= test_1.sample(0.01,200000)
  #hashing 'user_id' and 'track_id' columns. New columns useridint and trackidint created on all dataframes
  train_df= train_df.withColumn("useridint",hash(col("user_id")))
  train_df= train_df.withColumn("trackidint",hash(col("track_id")))
  val_df= val_df.withColumn("useridint",hash(col("user_id")))
  val_df= val_df.withColumn("trackidint",hash(col("track_id")))
  test_df= test_df.withColumn("useridint",hash(col("user_id")))
  test_df= test_df.withColumn("trackidint",hash(col("track_id")))
 #repartiting the dataframes
  train_df1= train_df.repartition(1000)
  test_df1= test_df.repartition(500)
  val_df1= val_df.repartition(500)
  spu=500 #songsperuser
  w=Window.partitionBy(val_df1['useridint']).orderBy(val_df1['count'].desc())#ordering dataset by count. Window function allows us to calc rank of a given row
  rank1= val_df1.select('*', F.rank().over(w).alias('rank')).filter(F.col('rank')<= spu).orderBy('useridint', 'rank').groupby('useridint').agg(F.collect_list('trackidint').alias('tracks'))
   #importing some essential pyspark libraries
  from pyspark.ml.recommendation import ALS
  from pyspark.ml.evaluation import RegressionEvaluator
  from pyspark.mllib.evaluation import RankingMetrics
  from pyspark.ml.tuning import ParamGridBuilder, TrainValidationSplit
  #creating our ALS model
  als = ALS(
         userCol="useridint", #user matrix
         itemCol="trackidint", #item matrix
         ratingCol="count", #ratings
         maxIter=10,
         seed= seed,
         nonnegative = False,
         implicitPrefs = True, #Implicit feedback
         coldStartStrategy="drop"
  )
  tolerance = 0.03
  ranks= [50,100,160, 200]
  regParams= [.15,0.2,0.25]
  errors = [[0]*len(ranks)]*len(regParams)
  models = [[0]*len(ranks)]*len(regParams)
  err = 0
  best_MAP = 0
  best_rank = -1
  i = 0
  for regParam in regParams:
    j = 0
    for rank in ranks:
      start_time = time.time()
      # Set the rank here:
      als.setParams(rank = rank, regParam = regParam)
    # Fit model on training data  with current parameters.
      model = als.fit(train_df1)
    # Run the model to create a prediction. Predict against the validation_df.
      predicted_rank=model.recommendForUserSubset(val_df1,500)
      end_time = time.time()
      diff_time=end_time - start_time #runtime
      print("The time taken is "+str(diff_time))
      predicted_rank = predicted_rank.join(rank1, "useridint", "inner").select('recommendations', 'tracks')
      labelrdd = predicted_rank.rdd.map(lambda x: ([row.trackidint for row in x[0]], x[1]),
                                            preservesPartitioning=True)
            # calculate MAP and Precision@K
      metrics = RankingMetrics(labelrdd)
      MAP=metrics.meanAveragePrecision
      prec_at=metrics.precisionAt(500)
      errors[i][j] = MAP
      models[i][j] = model
      print('For rank %s, regularization parameter %s the MAP is %s' % (rank, regParam, MAP))
      #Checking if MAP for this set of params is greater than prev MAP, if so, this set is the best set now
      if MAP > best_MAP:
         min_error = MAP
         best_params = [i,j]
      j += 1
    i += 1
  als.setRegParam(regParams[best_params[0]]) #setting best value for rank to model
  als.setRank(ranks[best_params[1]]) #setting best model for regParam to model
  print ('The best model was trained with regularization parameter %s' % regParams[best_params[0]])
  print ('The best model was trained with rank %s' % ranks[best_params[1]])
  my_model = models[best_params[0]][best_params[1]]
  #predicting on test data with best value for params
  final_predict= my_model.transform(test_df1)
  final_predict.show()

  #RMSE = evaluator.evaluate(test_predictions) if RMSE score is wanted
if __name__ == '__main__':
  spark = SparkSession.builder.appName('part2').getOrCreate()
  file_path1= sys.argv[1]
  file_path2= sys.argv[2]
  file_path3= sys.argv[3]
  main(spark,file_path1, file_path2, file_path3)
© 2021 GitHub, Inc.
Terms
Privacy
Security
Status
Docs
Contact GitHub
Pricing
API
Training
Blog
About
