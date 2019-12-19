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
import org.apache.spark.sql.SaveMode

object SampleJob5 extends SparkApp(name="Sample Job 5 1 (Without ML)", master="local[*]",
  conf=Seq(
    ("spark.extraListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.queryExecutionListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.streaming.streamingQueryListeners", "com.hortonworks.spark.atlas.SparkAtlasStreamingQueryEventTracker")
  )
) {

  // Initializing library to hook up to Apache Spark
//  spark.sparkContext.addFile("src/main/resources/atlas-application.properties")

  // A business logic of a spark job ...


  val inputDf = spark.read
    .option("header", "true")
    .option("inferSchema", "true")
    .csv("data/input/batch/ml/iris.csv")

  val transformedDf = inputDf.select($"sepal-length-in-cm", $"sepal-width-in-cm")
    .filter($"sepal-length-in-cm" > 5.0)

  transformedDf.show()
  transformedDf.write.mode(SaveMode.Overwrite)
    .csv("data/results/batch/job5_results/output")
}
