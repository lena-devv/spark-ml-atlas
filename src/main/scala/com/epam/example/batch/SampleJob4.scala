/*
 * Copyright 2017 ABSA Group Limited
 *
 * Licensed under the Apache License, Version 2.0 (the "License");
 * you may not use this file except in compliance with the License.
 * You may obtain a copy of the License at
 *
 *     http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing, software
 * distributed under the License is distributed on an "AS IS" BASIS,
 * WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
 * See the License for the specific language governing permissions and
 * limitations under the License.
 */

package com.epam.example.batch

import com.epam.example.SparkApp
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.classification.{LogisticRegression, OneVsRest, OneVsRestModel}
import org.apache.spark.ml.evaluation.MulticlassClassificationEvaluator
import org.apache.spark.ml.feature.{StringIndexer, VectorAssembler}
import org.apache.spark.sql.{DataFrame, SaveMode}
import org.apache.spark.sql.expressions.UserDefinedFunction
import org.apache.spark.sql.functions._

object SampleJob4 extends SparkApp("Sample Job 4 2 Spark ML",
  conf=Seq(
    ("spark.extraListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.queryExecutionListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.streaming.streamingQueryListeners", "com.hortonworks.spark.atlas.SparkAtlasStreamingQueryEventTracker")
  )
) {

  // Initializing library to hook up to Apache Spark

  spark.sparkContext.addFile("src/main/resources/atlas-application.properties")

  // A business logic of a spark job ...

  val targetToIntFunc = createTargetMappingUdf()
  val featuresColCreator = initFeaturesColumnCreator()
  val labelColIndexer = initLabelColumnIndexer("target")


  val inputDf = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/input/batch/ml/iris.csv")

  val df = prepareDf(inputDf, featuresColCreator, labelColIndexer)
  val Array(trainDf, testDf) = df.randomSplit(Array(0.8, 0.2))
  trainDf.printSchema()
  trainDf.show(15)
  testDf.show(15)

  val classifier = new LogisticRegression()
    .setMaxIter(20)
    .setTol(1E-6)
    .setFitIntercept(true)
  val ovrClassifier = new OneVsRest().setClassifier(classifier)

  val pipeline = new Pipeline().setStages(Array(ovrClassifier))
  val oneVsRestModel = ovrClassifier.fit(trainDf)

  val modelPath = "data/results/batch/job4_results/model"
  oneVsRestModel.write.overwrite().save(modelPath)

  val pipelinePath = "data/results/batch/job4_results/pipeline"
  pipeline.write.overwrite().save(pipelinePath)

  val loadedOvrModel = OneVsRestModel.load(modelPath)
  val predictions = loadedOvrModel.transform(testDf)
  predictions.show(15)

  // compute the classification error on test data
  val evaluator = new MulticlassClassificationEvaluator().setMetricName("accuracy")
  val accuracy = evaluator.evaluate(predictions)
  println(s"Test Error = ${1 - accuracy}")

  predictions.write.mode(SaveMode.Overwrite)
    .json("data/results/batch/job4_results/predictions")


  def initFeaturesColumnCreator(): VectorAssembler = {
    initFeaturesColumnCreator(Array(
      "sepal-length-in-cm", "sepal-width-in-cm", "petal-length-in-cm", "petal-width-in-cm"
    ))
  }

  def initFeaturesColumnCreator(columns: Array[String]): VectorAssembler = {
    new VectorAssembler()
      .setInputCols(columns)
      .setOutputCol("features")
  }

  def initLabelColumnIndexer(targetColumn: String): StringIndexer = {
    new StringIndexer()
      .setInputCol(targetColumn)
      .setOutputCol("label")
  }

  def createTargetMappingUdf(): UserDefinedFunction = {
    udf[Integer, String](target => {
      if (target.equals("Iris-setosa")) {
        1
      } else if (target.equals("Iris-versicolor")) {
        2
      } else {
        3
      }
    })
  }

  def prepareDf(inputDf: DataFrame, featuresColCreator: VectorAssembler, labelIndexer: StringIndexer): DataFrame = {
    var df = inputDf.withColumn("target", targetToIntFunc($"target-iris-type"))
    df = featuresColCreator.transform(df)
    val featuresWithLabelDf = labelIndexer.fit(df).transform(df)

    featuresWithLabelDf
  }
}
