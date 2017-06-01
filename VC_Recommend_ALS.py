#!/usr/bin/env python

import sys
import itertools
from math import sqrt
from operator import add
from os.path import join, isfile, dirname
from pyspark import SparkConf, SparkContext
from pyspark.mllib.recommendation import ALS
from pyspark.mllib.linalg import SparseVector
from pyspark import SparkConf, SparkContext
from pyspark.mllib.linalg import Vectors
from pyspark.mllib.recommendation import ALS, MatrixFactorizationModel, Rating
#####################################################################################
def parseRow(line):
    line_split = line.split(';')
    VC_id = int(line_split[0])
    VC_Investments = int(line_split[1])
    # print (str(VC_id), str(VC_Investments))
    return (VC_id , VC_Investments, 1.0)
#####################################################################################
""" Compute RMSE (Root Mean Squared Error)."""
def computeRmse(model, data, n):
    try: # print data.collect()
        print "Data:", data.take(2)
        print "Model:", type(model)
        predictions = model.predictAll(data.map(lambda x: (x[0], x[1])))
        print "Predictions:", predictions.take(2)

        predictionsAndRatings = predictions.map(lambda x: ((x[0], x[1]), x[2])) \
                                           .join(data.map(lambda x: ((x[0], x[1]), x[2]))) \
                                           .values()


        return sqrt(predictionsAndRatings.map(lambda x: (x[0] - x[1]) ** 2).reduce(add) / float(n))
    except Exception, e:
        print e        
    return 
#####################################################################################
if __name__ == "__main__":
    if (len(sys.argv) != 1):
        print "Usage: /path/to/spark/bin/spark-submit --driver-memory 2g " + \
              "VC_Recommend_ALS"
        sys.exit(1)

    # set up environment
    conf = SparkConf() \
        .setAppName("VC_ALS") \
        .set("spark.executor.memory", "2g") \
        .set("spark.driver.memory",  "2g") \
        .set("spark.driver.maxResultSize","2g")
    
    sc = SparkContext(conf=conf)
    #All VC ids along with their investments
    all_Investments_RDD = sc.textFile('VC_Investments.txt').map(parseRow)
    allInvestmentList = all_Investments_RDD.map(lambda x: x[1]).collect()

    myVC_rating = sc.textFile('My_VC.txt').map(parseRow)
    # print all_Investments_RDD.take(50)
    sc.setCheckpointDir('checkpoint/')
    

    #Creates an RDD: [Rating(user=0, product=41, rating=1.0), Rating(user=1, product=84, rating=1.0), ...]
    ratings = all_Investments_RDD.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))
    myVC_rating_RDD = myVC_rating.map(lambda l: Rating(int(l[0]), int(l[1]), float(l[2])))

    numInvestments = ratings.count()
    numVCs = ratings.map(lambda r: r[0]).distinct().count()
    numCompanies = ratings.map(lambda r: r[1]).distinct().count()

    print "Got %i investments from %i VC ." % (numCompanies, numVCs)

    numPartitions = 4
    # print ratings
    training = ratings.filter(lambda x: (x[0]%10) < 6) \
      .union(myVC_rating_RDD) \
      .repartition(numPartitions) \
      .cache()
    # training = training.map(lambda x: (x[0], x[1],x[2])) \

    # print type(training)
    # print training.take(5)

    validation = ratings.filter(lambda x: (x[0]%10) >= 6 and (x[0]%10 < 8)) \
      .repartition(numPartitions) \
      .cache()

    # print validation.take(5)

    test = ratings.filter(lambda x: (x[0]%10) >= 8).cache()
    # print test.take(5)

    numTraining = training.count()
    numValidation = validation.count()
    numTest = test.count()

    print "Training: %d, validation: %d, test: %d" % (numTraining, numValidation, numTest)
    

    ranks = [12]
    lambdas = [0.1, 10.0]
    numIters = 20
    bestModel = None
    bestValidationRmse = 10000
    bestRank = 0
    bestLambda = -1.0
    bestNumIter = 20

    ALS.checkpointInterval = 2
    # print "before for loops"
    for rank, lmbda in itertools.product(ranks, lambdas):
        # print ("Made it before training")
        print training.take(2)
        # print rank,
        # print numIter,
        # print lmbda
        model = ALS.train(training, rank, bestNumIter, lmbda)        

        print ("before the validationRMSE")
        validationRmse = computeRmse(model, training, numValidation)        
        print type(validationRmse)
        print "RMSE (validation) = %f for the model trained with " % validationRmse + \
              "rank = %d, lambda = %.1f, and numIter = %d." % (rank, lmbda, bestNumIter)
        if (validationRmse < bestValidationRmse):
            bestModel = model            
            bestValidationRmse = validationRmse
            bestRank = rank
            bestNumIter = numIters
    

    print "bestMode", type(bestModel)
    print "test", type(test)
    print "numTest", type(numTest)
    testRmse = computeRmse(bestModel, training, numTest)
    print "testRMSE", type(testRmse)
    # print testRmse
    # evaluate the best model on the test set
    print "The best model was trained with rank = %d and lambda = %.1f, " % (bestRank, bestLambda) \
      + "and numIter = %d, and its RMSE on the test set is %f." % (bestNumIter, testRmse)


    specific_vc = sc.textFile('My_VC.txt').map(parseRow)
    specific_vc_ids = specific_vc.map(lambda x: x[1]).collect()
    candidates = sc.parallelize([v for v in allInvestmentList if v not in specific_vc_ids])
    predictions = bestModel.predictAll(candidates.map(lambda x: (0, x))).collect()
    recommendations = sorted(predictions, key=lambda x: x[2], reverse=True)[:50]

    print "Companies recommended for you:"
    for i in xrange(len(recommendations)):
        print ("%2d: %s" % (i + 1, allInvestmentList[recommendations[i][1]])).encode('ascii', 'ignore')
    # rank = 3
    # numIterations = 20
    # model = ALS.train(ratings, rank, numIterations)

    # computeRmse(model, ratings, rank)

    # # Evaluate the model on training data
    # testdata = ratings.map(lambda p: (p[0], p[1]))
    # predictions = model.predictAll(testdata).map(lambda r: ((r[0], r[1]), r[2]))
    # ratesAndPreds = ratings.map(lambda r: ((r[0], r[1]), r[2])).join(predictions)
    # MSE = ratesAndPreds.map(lambda r: (r[1][0] - r[1][1])**2).mean()
    # print("Mean Squared Error = " + str(MSE))

    # # Save and load model
    # model.save(sc, "/Users/Salar_Hajimirsadeghi/Desktop/VC_als")
    # sameModel = MatrixFactorizationModel.load(sc, "Users/Salar_Hajimirsadeghi/Desktop/VC_als")


    sc.stop()
