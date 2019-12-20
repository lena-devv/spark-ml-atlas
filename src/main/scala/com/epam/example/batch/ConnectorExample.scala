package com.epam.example.batch

import com.epam.example.SparkApp
import org.apache.spark.ml.{Pipeline, PipelineModel}
import org.apache.spark.ml.feature.{StopWordsRemover, Tokenizer}
import org.apache.spark.sql.functions._

object ConnectorExample extends SparkApp(name="Connector Example Job with Pipeline", master="local[*]",
  conf=Seq(
    ("spark.extraListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.queryExecutionListeners", "com.hortonworks.spark.atlas.SparkAtlasEventTracker"),
    ("spark.sql.streaming.streamingQueryListeners", "com.hortonworks.spark.atlas.SparkAtlasStreamingQueryEventTracker")
  )
) {

  val trainingDf = spark.read
    .text("data/input/batch/ml/GUTINDEX-part.txt")
    .select($"value" as "text")
    .select(trim($"text") as "text")
    .filter(not($"text".isin("")))

  trainingDf.createOrReplaceTempView("training_table")
  val training = spark.sql("select * from training_table")

  training.show()
  training.printSchema()

  // Configure an ML pipeline, which consists of three stages: tokenizer, remover.
  val tokenizer = new Tokenizer().setInputCol("text").setOutputCol("words")

  val remover = new StopWordsRemover().setInputCol("words").setOutputCol("filtered")

  val pipeline = new Pipeline().setStages(Array(tokenizer, remover))

  val model = pipeline.fit(training)

  val pipelineDir = "data/results/job_example/pipeline_streaming_dir"

  val modelDir = "data/results/job_example/model_streaming_dir"

  pipeline.write.overwrite().save(pipelineDir)

  model.write.overwrite().save(modelDir)
}
