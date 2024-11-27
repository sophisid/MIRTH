import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.ml.clustering.{KMeans, KMeansModel}
import org.apache.spark.sql.expressions.Window
import scala.math._

object Clustering {

  // Function to calculate the ideal number of hash tables
  def calculateNumHashTables(similarityThreshold: Double, desiredCollisionProbability: Double): Int = {
    require(similarityThreshold > 0.0 && similarityThreshold < 1.0,
      "similarityThreshold must be between 0 and 1 (exclusive).")
    require(desiredCollisionProbability > 0.0 && desiredCollisionProbability < 1.0,
      "desiredCollisionProbability must be between 0 and 1 (exclusive).")

    val numerator = scala.math.log(1 - desiredCollisionProbability)
    val denominator = scala.math.log(similarityThreshold)

    val b = numerator / denominator

    // Round up to the nearest integer
    val numHashTables = scala.math.ceil(b).toInt

    numHashTables
  }

  // ===========================
  // Non-Incremental LSH
  // ===========================
  def performLSHClustering(df: DataFrame): DataFrame = {
    // Assemble the binary features into a vector
    val assembler = new VectorAssembler()
      .setInputCols(df.columns.filterNot(_ == "_nodeId"))
      .setOutputCol("features")

    val featureDF = assembler.transform(df)

    // Set your desired similarity threshold and collision probability
    val similarityThreshold = 0.8  // Adjust as needed
    val desiredCollisionProbability = 0.9  // Adjust as needed

    // Calculate the ideal number of hash tables
    val numHashTables = calculateNumHashTables(similarityThreshold, desiredCollisionProbability)

    println(s"Using numHashTables: $numHashTables")

    // Apply MinHash LSH
    val mh = new MinHashLSH()
      .setNumHashTables(numHashTables)
      .setInputCol("features")
      .setOutputCol("hashes")
      .setSeed(12345L)  // Set seed for reproducibility

    val model = mh.fit(featureDF)
    val lshDF = model.transform(featureDF)

    lshDF
  }

  // ===========================
  // Incremental LSH
  // ===========================
  def performLSHClusteringIncremental(df: DataFrame, increment: Int): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    // Assemble the binary features into a vector
    val assembler = new VectorAssembler()
      .setInputCols(df.columns.filterNot(_ == "_nodeId"))
      .setOutputCol("features")

    val featureDF = assembler.transform(df)

    // Assign consecutive row numbers for reliable chunking
    val window = Window.orderBy(monotonically_increasing_id())
    val dfWithRowNum = featureDF.withColumn("row_num", row_number().over(window))

    val totalRows = dfWithRowNum.count()
    val iterations = math.ceil(totalRows.toDouble / increment).toInt

    var cumulativeData = spark.emptyDataFrame
    var lshDF: DataFrame = spark.emptyDataFrame  // Initialize lshDF

    // Use a fixed MinHashLSH model parameters
    val similarityThreshold = 0.8  // Adjust as needed
    val desiredCollisionProbability = 0.9  // Adjust as needed
    val numHashTables = calculateNumHashTables(similarityThreshold, desiredCollisionProbability)

    val mh = new MinHashLSH()
      .setNumHashTables(numHashTables)
      .setInputCol("features")
      .setOutputCol("hashes")
      .setSeed(12345L)  // Set seed for reproducibility

    var model: MinHashLSHModel = null

    for (i <- 1 to iterations) {
      val startRow = (i - 1) * increment + 1
      val endRow = i * increment

      val chunkDF = dfWithRowNum
        .filter($"row_num" >= startRow && $"row_num" <= endRow)
        .drop("row_num")

      // Update cumulative data
      cumulativeData = if (cumulativeData.isEmpty) chunkDF else cumulativeData.union(chunkDF)

      println(s"Iteration $i/$iterations")

      // Fit MinHashLSH model on cumulative data
      model = mh.fit(cumulativeData)
      lshDF = model.transform(cumulativeData)  // Update lshDF

      // Print intermediate results
      println(s"\nAfter processing chunk $i/$iterations:")
      val (patterns, nodeIdToClusterLabel) = createPatternsFromLSHClusters(lshDF)
      // You can evaluate clustering here if needed
      // evaluateClustering(lshDF, nodeIdToClusterLabel)
    }

    lshDF  // Return lshDF instead of cumulativeData
  }

  // ===========================
  // Non-Incremental K-Means
  // ===========================
  def performKMeansClustering(df: DataFrame, k: Int): DataFrame = {
    val featureDF = if (df.columns.contains("features")) {
      df
    } else {
      val assembler = new VectorAssembler()
        .setInputCols(df.columns.filterNot(_ == "_nodeId"))
        .setOutputCol("features")

      assembler.transform(df)
    }

    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L) // For reproducibility
      .setFeaturesCol("features")
      .setPredictionCol("clusterLabel")

    val model = kmeans.fit(featureDF)
    val predictions = model.transform(featureDF)
    predictions
  }

  // ===========================
  // Incremental K-Means
  // ===========================
  def performKMeansClusteringIncremental(df: DataFrame, k: Int, increment: Int): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    // Check if 'features' column already exists
    val featureDF = if (df.columns.contains("features")) {
      df
    } else {
      // Assemble the binary features into a vector
      val assembler = new VectorAssembler()
        .setInputCols(df.columns.filterNot(col => col == "_nodeId" || col == "features"))
        .setOutputCol("features")
      assembler.transform(df)
    }

    // Assign consecutive row numbers for reliable chunking
    val window = Window.orderBy(monotonically_increasing_id())
    val dfWithRowNum = featureDF.withColumn("row_num", row_number().over(window))

    val totalRows = dfWithRowNum.count()
    val iterations = math.ceil(totalRows.toDouble / increment).toInt

    var cumulativeData = spark.emptyDataFrame
    var predictions: DataFrame = spark.emptyDataFrame  // Initialize predictions

    var model: KMeansModel = null

    for (i <- 1 to iterations) {
      val startRow = (i - 1) * increment + 1
      val endRow = i * increment

      val chunkDF = dfWithRowNum
        .filter($"row_num" >= startRow && $"row_num" <= endRow)
        .drop("row_num")

      // Update cumulative data
      cumulativeData = if (cumulativeData.isEmpty) chunkDF else cumulativeData.union(chunkDF)

      println(s"Iteration $i/$iterations")

      // Fit K-Means model on cumulative data
      val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
        .setFeaturesCol("features")
        .setPredictionCol("clusterLabel")

      model = kmeans.fit(cumulativeData)
      predictions = model.transform(cumulativeData)  // Update predictions

      // Print intermediate results
      println(s"\nAfter processing chunk $i/$iterations:")
      val (patterns, nodeIdToClusterLabel) = createPatternsFromKMeansClusters(predictions)
      // You can evaluate clustering here if needed
      // evaluateClustering(predictions, nodeIdToClusterLabel)
    }

    predictions  // Return predictions instead of cumulativeData
  }

  // ===========================
  // Create Patterns from LSH Clusters
  // ===========================
  def createPatternsFromLSHClusters(df: DataFrame): (Array[Pattern], Map[Long, String]) = {
    val spark = df.sparkSession
    import spark.implicits._

    // Exclude certain columns
    val excludeCols = Set("_nodeId", "features", "hashes", "hashKey")

    // Create a hashKey column by converting the hashes to a string
    val hashesToString = udf((hashes: Seq[Vector]) => {
      hashes.map(_.toArray.mkString("_")).mkString("_")
    })

    val dfWithHashKey = df.withColumn("hashKey", hashesToString(col("hashes")))

    // List of property columns
    val propertyCols = df.columns.filterNot(colName => excludeCols.contains(colName))

    // Build aggregation expressions for all property columns
    val aggExprs = List(
      collect_list(col("_nodeId")).alias("nodeIds"),
      count("*").alias("clusterSize")
    ) ++ propertyCols.map(colName => sum(col(colName)).alias(colName))

    // Perform groupBy and aggregate in one step
    val clustersAggDF = dfWithHashKey.groupBy("hashKey")
      .agg(aggExprs.head, aggExprs.tail: _*)

    // Map node IDs to cluster labels
    val nodeIdToClusterLabel = clustersAggDF.select("nodeIds", "hashKey")
      .as[(Seq[Long], String)]
      .flatMap { case (nodeIds, hashKey) =>
        val label = s"Cluster_$hashKey"
        nodeIds.map(id => id -> label)
      }.collect().toMap

    // Now, process clustersAggDF to create patterns
    val patterns = clustersAggDF.rdd.map { row =>
      val hashKey = row.getAs[String]("hashKey")
      val nodeIds = row.getAs[Seq[Long]]("nodeIds")
      val clusterSize = row.getAs[Long]("clusterSize")

      // Find common properties where sum equals clusterSize
      val commonProperties = propertyCols.filter { colName =>
        val colSum = row.getAs[Long](colName)
        colSum == clusterSize
      }

      // Create a Node with the common properties
      val node = Node(
        label = s"Cluster_$hashKey",
        properties = commonProperties.map(prop => prop -> 1).toMap,
        patternId = s"Pattern_$hashKey"
      )

      // Create a Pattern with the node
      val pattern = new Pattern(nodes = List(node))

      pattern
    }.collect()

    println(s"Total patterns created: ${patterns.length}")
    println("Sample patterns:")
    patterns.take(5).foreach(pattern => println(pattern.toString))

    // Return patterns and nodeIdToClusterLabel mapping
    (patterns, nodeIdToClusterLabel)
  }

  // ===========================
  // Create Patterns from K-Means Clusters
  // ===========================
  def createPatternsFromKMeansClusters(df: DataFrame): (Array[Pattern], Map[Long, String]) = {
    val spark = df.sparkSession
    import spark.implicits._

    // Exclude certain columns
    val excludeCols = Set("_nodeId", "features", "clusterLabel")

    // List of property columns
    val propertyCols = df.columns.filterNot(colName => excludeCols.contains(colName))

    // Build aggregation expressions for all property columns
    val aggExprs = List(
      collect_list(col("_nodeId")).alias("nodeIds"),
      count("*").alias("clusterSize")
    ) ++ propertyCols.map(colName => sum(col(colName)).alias(colName))

    // Perform groupBy and aggregate
    val clustersAggDF = df.groupBy("clusterLabel")
      .agg(aggExprs.head, aggExprs.tail: _*)

    // Map node IDs to cluster labels
    val nodeIdToClusterLabel = df.select("_nodeId", "clusterLabel")
      .as[(Long, Int)]
      .map { case (nodeId, clusterLabel) =>
        nodeId -> s"Cluster_$clusterLabel"
      }.collect().toMap

    // Process clustersAggDF to create patterns
    val patterns = clustersAggDF.rdd.map { row =>
      val clusterLabel = row.getAs[Int]("clusterLabel")
      val nodeIds = row.getAs[Seq[Long]]("nodeIds")
      val clusterSize = row.getAs[Long]("clusterSize")

      // Find common properties where sum equals clusterSize
      val commonProperties = propertyCols.filter { colName =>
        val colSum = row.getAs[Long](colName)
        colSum == clusterSize
      }

      // Create a Node with the common properties
      val node = Node(
        label = s"Cluster_$clusterLabel",
        properties = commonProperties.map(prop => prop -> 1).toMap,
        patternId = s"Pattern_$clusterLabel"
      )

      // Create a Pattern with the node
      val pattern = new Pattern(nodes = List(node))
      pattern
    }.collect()

    println(s"Total patterns created: ${patterns.length}")
    println("Sample patterns:")
    patterns.take(5).foreach(pattern => println(pattern.toString))

    // Return patterns and nodeIdToClusterLabel mapping
    (patterns, nodeIdToClusterLabel)
  }

  // ===========================
  // Create Edges from Relationships
  // ===========================
  def createEdgesFromRelationships(
      relationshipsDF: DataFrame,
      nodeIdToClusterLabel: Map[Long, String]
  ): Array[Edge] = {
    val spark = relationshipsDF.sparkSession
    import spark.implicits._

    // Broadcast the nodeIdToClusterLabel mapping for efficiency
    val nodeIdToClusterLabelBroadcast = spark.sparkContext.broadcast(nodeIdToClusterLabel)

    // Function to extract node type from the cluster label
    def extractNodeType(label: String): String = label.split("_").drop(1).mkString("_")

    // Map relationships to edges
    val edges = relationshipsDF.rdd.flatMap { row =>
      val srcId = row.getAs[Long]("srcId")
      val dstId = row.getAs[Long]("dstId")
      val relationshipType = row.getAs[String]("relationshipType")
      val properties = row.getAs[Map[String, Any]]("properties")

      val clusterLabelSrc = nodeIdToClusterLabelBroadcast.value.get(srcId)
      val clusterLabelDst = nodeIdToClusterLabelBroadcast.value.get(dstId)

      // Only create edge if both nodes have cluster labels and are in different clusters
      for {
        startLabel <- clusterLabelSrc
        endLabel <- clusterLabelDst
        if startLabel != endLabel
      } yield {
        // Extract the types from labels for both start and end nodes
        val startType = extractNodeType(startLabel)
        val endType = extractNodeType(endLabel)

        val startNode = Node(label = startLabel, properties = Map.empty, patternId = s"Pattern_$startLabel")
        val endNode = Node(label = endLabel, properties = Map.empty, patternId = s"Pattern_$endLabel")

        Edge(
          startNode = startNode,
          relationshipType = relationshipType,
          endNode = endNode,
          properties = properties,
          patternId = s"Pattern_${startType}_to_${endType}"
        )
      }
    }

    // Collect unique edges by (startType, relationshipType, endType)
    val uniqueEdges = edges
      .map(edge => ((extractNodeType(edge.startNode.label), edge.relationshipType, extractNodeType(edge.endNode.label)), edge))
      .reduceByKey((edge1, _) => edge1) // Keep one edge per unique key
      .values
      .collect()

    uniqueEdges
  }

  // ===========================
  // Integrate Edges into Patterns
  // ===========================
  def integrateEdgesIntoPatterns(
      edges: Array[Edge],
      existingPatterns: Array[Pattern]
  ): Array[Pattern] = {
    // Map cluster labels to patterns
    val clusterLabelToPattern = existingPatterns.map(pattern => pattern.nodes.head.label -> pattern).toMap

    // For any new clusters, add patterns
    val patternsMap = collection.mutable.Map(clusterLabelToPattern.toSeq: _*)

    edges.foreach { edge =>
      // Get or create the pattern for the start node
      val startPattern = patternsMap.getOrElseUpdate(edge.startNode.label, {
        val newPattern = new Pattern()
        newPattern.addNode(edge.startNode)
        newPattern
      })

      // Add the end node to the start pattern if not present
      if (!startPattern.nodes.exists(_.label == edge.endNode.label)) {
        startPattern.addNode(edge.endNode)
      }

      // Add the edge to the start pattern
      startPattern.addEdge(edge)
    }

    patternsMap.values.toArray
  }

  // ===========================
  // Evaluate Clustering
  // ===========================
  def evaluateClustering(nodesDF: DataFrame, nodeIdToClusterLabel: Map[Long, String]): Unit = {
    val spark = nodesDF.sparkSession
    import spark.implicits._

    // Prepare data for evaluation
    val predictedLabelsDF = nodeIdToClusterLabel.toSeq.toDF("_nodeId", "predictedClusterLabel")
    val nodesWithLabelsDF = nodesDF.select($"_nodeId", $"_labels")
    val evaluationDF = nodesWithLabelsDF.join(predictedLabelsDF, "_nodeId")

    evaluationDF.cache()
    evaluationDF.count() // Trigger caching

    // Compute metrics using the existing method
    ClusteringEvaluation.computeMetricsWithoutPairwise(evaluationDF)
  }

  // ===========================
  // Compute Pattern Similarity
  // ===========================
  def computePatternSimilarity(pattern1: Pattern, pattern2: Pattern): Double = {
    val properties1 = pattern1.nodes.flatMap(_.properties.keys).toSet
    val properties2 = pattern2.nodes.flatMap(_.properties.keys).toSet
    val intersectionSize = properties1.intersect(properties2).size
    val unionSize = properties1.union(properties2).size
    if (unionSize == 0) 0.0 else intersectionSize.toDouble / unionSize
  }

  // ===========================
  // Merge Similar Patterns
  // ===========================
  def mergeSimilarPatterns(patterns: Array[Pattern], similarityThreshold: Double): Array[Pattern] = {
    val mergedPatterns = mutable.ArrayBuffer[Pattern]()
    val visited = mutable.Set[Int]() // Indices of patterns already merged

    for (i <- patterns.indices) {
      if (!visited.contains(i)) {
        var basePattern = patterns(i)
        val similarIndices = (i + 1 until patterns.length).filter { j =>
          if (!visited.contains(j)) {
            val similarity = computePatternSimilarity(basePattern, patterns(j))
            similarity >= similarityThreshold
          } else {
            false
          }
        }

        // Merge similar patterns
        similarIndices.foreach { j =>
          basePattern = mergeTwoPatterns(basePattern, patterns(j))
          visited.add(j)
        }

        mergedPatterns += basePattern
        visited.add(i)
      }
    }

    mergedPatterns.toArray
  }
  // Helper function to merge two patterns
  def mergeTwoPatterns(pattern1: Pattern, pattern2: Pattern): Pattern = {
    // Merge nodes
    val nodesMap = mutable.Map[String, Node]()
    (pattern1.nodes ++ pattern2.nodes).foreach { node =>
      if (nodesMap.contains(node.label)) {
        // Merge properties
        val existingNode = nodesMap(node.label)
        val mergedProperties = mergeProperties(existingNode.properties, node.properties)
        nodesMap(node.label) = existingNode.copy(properties = mergedProperties)
      } else {
        nodesMap(node.label) = node
      }
    }

    // Merge edges
    val edges = (pattern1.edges ++ pattern2.edges).distinct

    // Create new pattern
    new Pattern(nodes = nodesMap.values.toList, edges = edges)
  }
  // Helper function to merge properties and mark extra ones as optional
  def mergeProperties(props1: Map[String, Any], props2: Map[String, Any]): Map[String, Any] = {
    val allKeys = props1.keySet.union(props2.keySet)
    allKeys.map { key =>
      val value = (props1.get(key), props2.get(key)) match {
        case (Some(v1), Some(v2)) => v1 // Common property
        case _ => "Optional" // Mark as optional
      }
      key -> value
    }.toMap
  }
   // ===========================
  // Update Node ID to Cluster Label After Merging
  // ===========================
  def updateNodeIdToClusterLabelAfterMerging(
      originalNodeIdToClusterLabel: Map[Long, String],
      originalPatterns: Array[Pattern],
      mergedPatterns: Array[Pattern]
  ): Map[Long, String] = {
    // Create a mapping from old pattern labels to new merged pattern labels
    val oldLabelToNewLabel = mutable.Map[String, String]()

    // For each merged pattern, find which original patterns were merged
    for (mergedPattern <- mergedPatterns) {
      val mergedNodeLabels = mergedPattern.nodes.map(_.label).toSet
      val matchingPatterns = originalPatterns.filter(p => mergedNodeLabels.exists(label => p.nodes.exists(_.label == label)))
      for (pattern <- matchingPatterns) {
        oldLabelToNewLabel(pattern.nodes.head.label) = mergedPattern.nodes.head.label
      }
    }

    // Update nodeIdToClusterLabel mapping
    val updatedNodeIdToClusterLabel = originalNodeIdToClusterLabel.map { case (nodeId, oldLabel) =>
      val newLabel = oldLabelToNewLabel.getOrElse(oldLabel, oldLabel)
      nodeId -> newLabel
    }

    updatedNodeIdToClusterLabel
  }
}
