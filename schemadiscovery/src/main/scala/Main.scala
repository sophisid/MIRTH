import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.neo4j.driver.{AuthTokens, GraphDatabase}
import java.time.Duration
import java.time.Instant


object Main {
  def main(args: Array[String]): Unit = {
    val programStartTime = System.nanoTime()
    val spark = SparkSession.builder()
      .appName("SchemaDiscoveryComparison")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    // Load all nodes with labels
    val dataLoadStartTime = System.nanoTime()
    val nodesDF = DataLoader.loadAllNodes(spark).cache()

    // Assign "NoLabel" to nodes with null or empty _labels
    val nodesWithLabelsDF = nodesDF
      .withColumn("_labels", when($"_labels".isNull || length(trim($"_labels")) === 0, "NoLabel").otherwise($"_labels"))
      .cache()

    // Load relationships
    val relationshipsDF = DataLoader.loadAllRelationships(spark).cache()
    val dataLoadEndTime = System.nanoTime()
    val dataLoadTime = (dataLoadEndTime - dataLoadStartTime) / 1e9d  // Convert to seconds
    println(f"\nTime taken for Data Loading and Preprocessing: $dataLoadTime%.2f seconds")


    // Argument for which clustering to run
    if (args.length == 0) {
      println("Please provide arguments: 'l' for LSH only, 'k' for K-Means only, 'b' for both.")
      sys.exit(1)
    }

    val mode = args(0).toLowerCase
    val isIncremental = args.contains("i") // Check if 'i' is among the arguments
    val increment = if (isIncremental) {
      val incrementIndex = args.indexOf("i") + 1
      if (args.length > incrementIndex) {
        args(incrementIndex).toInt // Get increment value from arguments
      } else {
        1000 // Default increment value
      }
    } else {
      0 // No incrementality
    }

    mode match {
      case "l" => {
        runLSHClustering(spark, nodesWithLabelsDF, relationshipsDF, increment)
      }
      case "k" => {
        runKMeansClustering(spark, nodesWithLabelsDF, relationshipsDF, increment)
      }
      case "b" => {
        // Run both LSH and K-Means Clustering
        runLSHClustering(spark, nodesWithLabelsDF, relationshipsDF, increment)
        runKMeansClustering(spark, nodesWithLabelsDF, relationshipsDF, increment)
      } // Missing closing brace added here
      case _ => {
        println("Invalid argument. Please provide 'l' for LSH only, 'k' for K-Means only, 'b' for both.")
        sys.exit(1)
      }
    }

    spark.stop()
    val programEndTime = System.nanoTime()
    val totalProgramTime = (programEndTime - programStartTime) / 1e9d  // Convert to seconds
    println(f"\nTotal Program Execution Time: $totalProgramTime%.2f seconds")
  }

  def evaluateClustering(nodesDF: DataFrame, nodeIdToClusterLabel: Map[Long, String]): Unit = {
    val spark = nodesDF.sparkSession
    import spark.implicits._

    // Prepare data for evaluation
    val predictedLabelsDF = nodeIdToClusterLabel.toSeq.toDF("_nodeId", "predictedClusterLabel")
    val nodesWithLabelsDF = nodesDF.select($"_nodeId".cast(LongType), $"_labels")
    val evaluationDF = nodesWithLabelsDF.join(predictedLabelsDF, "_nodeId")

    evaluationDF.cache()
    evaluationDF.count() // Trigger caching

    // Compute metrics using the existing method
    ClusteringEvaluation.computeMetricsWithoutPairwise(evaluationDF)
  }

  def runLSHClustering(spark: SparkSession, nodesDF: DataFrame, relationshipsDF: DataFrame, increment: Int): Unit = {
    import spark.implicits._
    val runtime = Runtime.getRuntime

    // Trigger garbage collection before measuring initial memory
    runtime.gc()
    val initialMemory = runtime.totalMemory() - runtime.freeMemory()
    val clusteringStartTime = System.nanoTime()

    // LSH Clustering Pipeline
    val binaryMatrixDF_LSH = DataProcessor.createBinaryMatrix(nodesDF).cache()
    val lshDF = if (increment > 0) {
      Clustering.performLSHClusteringIncremental(binaryMatrixDF_LSH, increment).cache()
    } else {
      Clustering.performLSHClustering(binaryMatrixDF_LSH).cache()
    }
    val (lshPatterns, lshNodeIdToClusterLabel) = Clustering.createPatternsFromLSHClusters(lshDF)
    val edges_LSH = Clustering.createEdgesFromRelationships(relationshipsDF, lshNodeIdToClusterLabel)
    val updatedPatterns_LSH = Clustering.integrateEdgesIntoPatterns(edges_LSH, lshPatterns)

    val clusteringEndTime = System.nanoTime()
    val clusteringTime = (clusteringEndTime - clusteringStartTime) / 1e9d  // Convert to seconds
    runtime.gc()
    val finalMemory = runtime.totalMemory() - runtime.freeMemory()

    // Calculate memory used during clustering in MB
    val memoryUsedMB = (finalMemory - initialMemory) / (1024 * 1024)

    println(f"\nTime taken for LSH Clustering: $clusteringTime%.2f seconds")
    println(f"Memory used for LSH Clustering: $memoryUsedMB%.2f MB")

    println("=== Evaluation Results for LSH Clustering ===")
    evaluateClustering(nodesDF, lshNodeIdToClusterLabel)

    val numNodes_LSH = lshNodeIdToClusterLabel.size
    val numEdges_LSH = edges_LSH.size
    val numPatterns_LSH = lshNodeIdToClusterLabel.values.toSet.size
    val types_LSH = lshNodeIdToClusterLabel.values.toSet

    println("=== LSH Clustering Results ===")
    println(s"Number of nodes assigned to clusters: $numNodes_LSH")
    println(s"Number of edges after clustering: $numEdges_LSH")
    println(s"Number of patterns (types) found: $numPatterns_LSH")
    println(s"Types found: ${types_LSH.mkString(", ")}")
  }

  def runKMeansClustering(spark: SparkSession, nodesDF: DataFrame, relationshipsDF: DataFrame, increment: Int): Unit = {
    import spark.implicits._
    val runtime = Runtime.getRuntime

    // Trigger garbage collection before measuring initial memory
    runtime.gc()
    val initialMemory = runtime.totalMemory() - runtime.freeMemory()


    val clusteringStartTime = System.nanoTime()

    // K-Means Clustering Pipeline
    val featureDF_KMeans = DataProcessor.assembleFeaturesForKMeans(nodesDF).cache()

    // Compute the number of distinct types/patterns in the Neo4j database
    val numTypes = nodesDF
      .withColumn("label_array", split($"_labels", ",")) // Adjust delimiter if necessary
      .withColumn("label_array", expr("filter(label_array, x -> x != '')")) // Remove empty strings from array
      .select(explode($"label_array").alias("label"))
      .select(trim($"label").alias("label"))
      .distinct()
      .count()
      .toInt

    val k = numTypes // Set k dynamically based on the number of types
    println(s"Number of distinct types/patterns in Neo4j (including 'NoLabel'): $numTypes")
    println(s"Setting k for K-Means clustering to: $k")

    val kmeansDF = if (increment > 0) {
      Clustering.performKMeansClusteringIncremental(featureDF_KMeans, k, increment).cache()
    } else {
      Clustering.performKMeansClustering(featureDF_KMeans, k).cache()
    }

    val (kmeansPatterns, kmeansNodeIdToClusterLabel) = Clustering.createPatternsFromKMeansClusters(kmeansDF)
    val edges_KMeans = Clustering.createEdgesFromRelationships(relationshipsDF, kmeansNodeIdToClusterLabel)
    val updatedPatterns_KMeans = Clustering.integrateEdgesIntoPatterns(edges_KMeans, kmeansPatterns)

    val clusteringEndTime = System.nanoTime()
    val clusteringTime = (clusteringEndTime - clusteringStartTime) / 1e9d  // Convert to seconds
    runtime.gc()
    val finalMemory = runtime.totalMemory() - runtime.freeMemory()

    // Calculate memory used during clustering in MB
    val memoryUsedMB = (finalMemory - initialMemory) / (1024 * 1024)

    println(f"\nTime taken for K-Means Clustering: $clusteringTime%.2f seconds")
    println(f"Memory used for K-Means Clustering: $memoryUsedMB%.2f MB")


    println("=== Evaluation Results for K-Means Clustering ===")
    evaluateClustering(nodesDF, kmeansNodeIdToClusterLabel)

    val numNodes_KMeans = kmeansNodeIdToClusterLabel.size
    val numEdges_KMeans = edges_KMeans.size
    val numPatterns_KMeans = kmeansNodeIdToClusterLabel.values.toSet.size
    val types_KMeans = kmeansNodeIdToClusterLabel.values.toSet

    println("=== K-Means Clustering Results ===")
    println(s"Number of nodes assigned to clusters: $numNodes_KMeans")
    println(s"Number of edges after clustering: $numEdges_KMeans")
    println(s"Number of patterns (types) found: $numPatterns_KMeans")
    println(s"Types found: ${types_KMeans.mkString(", ")}")
  }
}
