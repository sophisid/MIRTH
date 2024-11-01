import org.apache.spark.sql.SparkSession
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import org.neo4j.driver.{AuthTokens, GraphDatabase}

object Main {
  def main(args: Array[String]): Unit = {
    val spark = SparkSession.builder()
      .appName("SchemaDiscoveryComparison")
      .master("local[*]")
      .config("spark.serializer", "org.apache.spark.serializer.JavaSerializer")
      .getOrCreate()

    spark.sparkContext.setLogLevel("ERROR")

    import spark.implicits._

    // Load all nodes with labels
    val nodesDF = DataLoader.loadAllNodes(spark).cache()

    // Assign "NoLabel" to nodes with null or empty _labels
    val nodesWithLabelsDF = nodesDF
      .withColumn("_labels", when($"_labels".isNull || length(trim($"_labels")) === 0, "NoLabel").otherwise($"_labels"))
      .cache()

    // Load relationships
    val relationshipsDF = DataLoader.loadAllRelationships(spark).cache()

    // Argument for which clustering to run
    if (args.length == 0) {
      println("Please provide an argument: 'l' for LSH only, 'k' for K-Means only, 'b' for both.")
      sys.exit(1)
    }
    val mode = args(0).toLowerCase

    mode match {
      case "l" => {
        // LSH Clustering Pipeline
        val binaryMatrixDF_LSH = DataProcessor.createBinaryMatrix(nodesWithLabelsDF).cache()
        val lshDF = Clustering.performLSHClustering(binaryMatrixDF_LSH).cache()
        val (lshPatterns, lshNodeIdToClusterLabel) = Clustering.createPatternsFromLSHClusters(lshDF)
        val edges_LSH = Clustering.createEdgesFromRelationships(relationshipsDF, lshNodeIdToClusterLabel)
        val updatedPatterns_LSH = Clustering.integrateEdgesIntoPatterns(edges_LSH, lshPatterns)
        println("=== Evaluation Results for LSH Clustering ===")
        evaluateClustering(nodesWithLabelsDF, lshNodeIdToClusterLabel)

        val numNodes_LSH = lshNodeIdToClusterLabel.size
        val numEdges_LSH = edges_LSH.size  // Use size instead of count()
        val numPatterns_LSH = lshNodeIdToClusterLabel.values.toSet.size
        val types_LSH = lshNodeIdToClusterLabel.values.toSet

        println("=== LSH Clustering Results ===")
        println(s"Number of nodes assigned to clusters: $numNodes_LSH")
        println(s"Number of edges after clustering: $numEdges_LSH")
        println(s"Number of patterns (types) found: $numPatterns_LSH")
        println(s"Types found: ${types_LSH.mkString(", ")}")
      }
      case "k" => {
        // K-Means Clustering Pipeline
        val featureDF_KMeans = DataProcessor.assembleFeaturesForKMeans(nodesWithLabelsDF).cache()

        // Compute the number of distinct types/patterns in the Neo4j database
        // val numTypes = nodesWithLabelsDF
        //   .withColumn("label_array", split($"_labels", ",")) // Adjust delimiter if necessary
        //   .withColumn("label_array", expr("filter(label_array, x -> x != '')")) // Remove empty strings from array
        //   .select(explode($"label_array").alias("label"))
        //   .select(trim($"label").alias("label"))
        //   .distinct()
        //   .count()
        //   .toInt
        val uri = "bolt://localhost:7687"
        val user = "neo4j"
        val password = "password"

        val driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))
        val session = driver.session()

        println("Loading all nodes from Neo4j")

        // Neo4j query to get the unique pattern count
        val result = session.run("MATCH (n) WITH labels(n) AS node_labels, keys(n) AS attributes RETURN COUNT(DISTINCT {node_labels: node_labels, attributes: attributes}) AS unique_pattern_count")

        // Extract the count as an integer from the result
        val numTypes = if (result.hasNext) result.next().get("unique_pattern_count").asInt() else 0

        val k = numTypes // Set k dynamically based on the number of types
        println(s"Number of distinct types/patterns in Neo4j (including 'NoLabel'): $numTypes")
        println(s"Setting k for K-Means clustering to: $k")

        val kmeansDF = Clustering.performKMeansClustering(featureDF_KMeans, k).cache()
        val (kmeansPatterns, kmeansNodeIdToClusterLabel) = Clustering.createPatternsFromKMeansClusters(kmeansDF)
        val edges_KMeans = Clustering.createEdgesFromRelationships(relationshipsDF, kmeansNodeIdToClusterLabel)
        val updatedPatterns_KMeans = Clustering.integrateEdgesIntoPatterns(edges_KMeans, kmeansPatterns)
        println("=== Evaluation Results for K-Means Clustering ===")
        evaluateClustering(nodesWithLabelsDF, kmeansNodeIdToClusterLabel)

        val numNodes_KMeans = kmeansNodeIdToClusterLabel.size
        val numEdges_KMeans = edges_KMeans.size  // Use size instead of count()
        val numPatterns_KMeans = kmeansNodeIdToClusterLabel.values.toSet.size
        val types_KMeans = kmeansNodeIdToClusterLabel.values.toSet

        println("=== K-Means Clustering Results ===")
        println(s"Number of nodes assigned to clusters: $numNodes_KMeans")
        println(s"Number of edges after clustering: $numEdges_KMeans")
        println(s"Number of patterns (types) found: $numPatterns_KMeans")
        println(s"Types found: ${types_KMeans.mkString(", ")}")
      }
      case "b" => {
        // Run both LSH and K-Means Clustering
        runLSHClustering(spark, nodesWithLabelsDF, relationshipsDF)
        runKMeansClustering(spark, nodesWithLabelsDF, relationshipsDF)
      }
      case _ => {
        println("Invalid argument. Please provide 'l' for LSH only, 'k' for K-Means only, 'b' for both.")
        sys.exit(1)
      }
    }

    spark.stop()
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

  def runLSHClustering(spark: SparkSession, nodesDF: DataFrame, relationshipsDF: DataFrame): Unit = {
    import spark.implicits._

    // LSH Clustering Pipeline
    val binaryMatrixDF_LSH = DataProcessor.createBinaryMatrix(nodesDF).cache()
    val lshDF = Clustering.performLSHClustering(binaryMatrixDF_LSH).cache()
    val (lshPatterns, lshNodeIdToClusterLabel) = Clustering.createPatternsFromLSHClusters(lshDF)
    val edges_LSH = Clustering.createEdgesFromRelationships(relationshipsDF, lshNodeIdToClusterLabel)
    val updatedPatterns_LSH = Clustering.integrateEdgesIntoPatterns(edges_LSH, lshPatterns)
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

  def runKMeansClustering(spark: SparkSession, nodesDF: DataFrame, relationshipsDF: DataFrame): Unit = {
    import spark.implicits._

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

    val kmeansDF = Clustering.performKMeansClustering(featureDF_KMeans, k).cache()
    val (kmeansPatterns, kmeansNodeIdToClusterLabel) = Clustering.createPatternsFromKMeansClusters(kmeansDF)
    val edges_KMeans = Clustering.createEdgesFromRelationships(relationshipsDF, kmeansNodeIdToClusterLabel)
    val updatedPatterns_KMeans = Clustering.integrateEdgesIntoPatterns(edges_KMeans, kmeansPatterns)
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
