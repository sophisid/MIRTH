import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.MinHashLSH
import org.apache.spark.ml.clustering.KMeans
import org.apache.spark.sql.expressions.Window
import scala.math._

object Clustering {

  // Function to calculate the ideal number of hash tables for LSH
  def calculateNumHashTables(similarityThreshold: Double, desiredCollisionProbability: Double): Int = {
    require(similarityThreshold > 0.0 && similarityThreshold < 1.0,
      "similarityThreshold must be between 0 and 1 (exclusive).")
    require(desiredCollisionProbability > 0.0 && desiredCollisionProbability < 1.0,
      "desiredCollisionProbability must be between 0 and 1 (exclusive).")

    val numerator = scala.math.log(1 - desiredCollisionProbability)
    val denominator = scala.math.log(1 - similarityThreshold)

    val b = numerator / denominator

    // Round up to the nearest integer
    val numHashTables = scala.math.ceil(b).toInt

    numHashTables
  }

  // Function to perform LSH clustering
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
    val numHashTablesCalculated = calculateNumHashTables(similarityThreshold, desiredCollisionProbability)

    // Adjust based on dataset size
    val datasetSize = df.count()
    val scalingFactor = math.log10(datasetSize)
    val numHashTablesAdjusted = (numHashTablesCalculated * scalingFactor).toInt

    // Set a minimum or maximum as needed
    val numHashTables = math.max(numHashTablesAdjusted, numHashTablesCalculated)

    println(s"Calculated numHashTables: $numHashTablesCalculated, Adjusted numHashTables: $numHashTables")

    // Apply MinHash LSH
    val mh = new MinHashLSH()
      .setNumHashTables(numHashTables)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = mh.fit(featureDF)
    val lshDF = model.transform(featureDF)

    lshDF
  }

  // Function to perform LSH clustering incrementally
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

    for (i <- 1 to iterations) {
      val startRow = (i - 1) * increment + 1
      val endRow = i * increment

      val chunkDF = dfWithRowNum
        .filter($"row_num" >= startRow && $"row_num" <= endRow)
        .drop("row_num")

      // Update cumulative data
      cumulativeData = if (cumulativeData.isEmpty) chunkDF else cumulativeData.union(chunkDF)

      // Define similarity threshold and desired collision probability
      val similarityThreshold = 0.8  // Adjust as needed
      val desiredCollisionProbability = 0.9  // Adjust as needed

      // Calculate the ideal number of hash tables
      val numHashTablesCalculated = calculateNumHashTables(similarityThreshold, desiredCollisionProbability)

      // Adjust based on cumulative dataset size
      val datasetSize = cumulativeData.count()
      val scalingFactor = math.log10(datasetSize + 1)  // Add 1 to avoid log(0)
      val numHashTablesAdjusted = (numHashTablesCalculated * scalingFactor).toInt

      // Set a minimum or maximum as needed
      val numHashTables = math.max(numHashTablesAdjusted, numHashTablesCalculated)

      println(s"Iteration $i/$iterations")
      println(s"Calculated numHashTables: $numHashTablesCalculated, Adjusted numHashTables: $numHashTables")

      // Apply MinHash LSH on cumulative data
      val mh = new MinHashLSH()
        .setNumHashTables(numHashTables)
        .setInputCol("features")
        .setOutputCol("hashes")

      val model = mh.fit(cumulativeData)
      lshDF = model.transform(cumulativeData)  // Update lshDF

      // Print intermediate results
      println(s"\nAfter processing chunk $i/$iterations:")
      val (patterns, nodeIdToClusterLabel) = createPatternsFromLSHClusters(lshDF)
      // Optionally evaluate clustering
      // evaluateClustering(lshDF, nodeIdToClusterLabel)
    }

    lshDF  // Return lshDF instead of cumulativeData
  }

  // Function to create patterns from LSH clusters
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
      .as[(Seq[String], String)]
      .flatMap { case (nodeIds, hashKey) =>
        val label = s"Cluster_$hashKey"
        nodeIds.map(id => id.toLong -> label)
      }.collect().toMap

    // Now, process clustersAggDF to create patterns
    val patterns = clustersAggDF.rdd.map { row =>
      val hashKey = row.getAs[String]("hashKey")
      val nodeIds = row.getAs[Seq[String]]("nodeIds")
      val clusterSize = row.getAs[Long]("clusterSize")

      // Find common properties where sum equals clusterSize
      val commonProperties = propertyCols.filter { colName =>
        val colSum = row.getAs[Any](colName) match {
          case n: Int => n.toLong
          case n: Long => n
          case n: Double => n.toLong
          case _ => 0L
        }
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

  // Function to create edges from relationships
  def createEdgesFromRelationships(
      relationshipsDF: DataFrame,
      nodeIdToClusterLabel: Map[Long, String]
  ): Array[Edge] = {
    val spark = relationshipsDF.sparkSession
    import spark.implicits._

    // Broadcast the nodeIdToClusterLabel mapping for efficiency
    val nodeIdToClusterLabelBroadcast = spark.sparkContext.broadcast(nodeIdToClusterLabel)

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
        val startNode = Node(label = startLabel, properties = Map.empty, patternId = s"Pattern_$startLabel")
        val endNode = Node(label = endLabel, properties = Map.empty, patternId = s"Pattern_$endLabel")

        Edge(
          startNode = startNode,
          relationshipType = relationshipType,
          endNode = endNode,
          properties = properties,
          patternId = s"Pattern_${startLabel}_to_${endLabel}"
        )
      }
    }

    // Collect unique edges by (startLabel, relationshipType, endLabel)
    val uniqueEdges = edges
      .map(edge => ((edge.startNode.label, edge.relationshipType, edge.endNode.label), edge))
      .reduceByKey((edge1, _) => edge1) // Keep one edge per unique key
      .values
      .collect()

    uniqueEdges
  }

  // Function to integrate edges into patterns
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

  // Function to perform K-Means clustering
  def performKMeansClustering(df: DataFrame, k: Int): DataFrame = {
    val kmeans = new KMeans()
      .setK(k)
      .setSeed(1L) // For reproducibility
      .setFeaturesCol("features")
      .setPredictionCol("clusterLabel")

    val model = kmeans.fit(df)
    val predictions = model.transform(df)
    predictions
  }

  // Function to perform K-Means clustering incrementally
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

    for (i <- 1 to iterations) {
      val startRow = (i - 1) * increment + 1
      val endRow = i * increment

      val chunkDF = dfWithRowNum
        .filter($"row_num" >= startRow && $"row_num" <= endRow)
        .drop("row_num")

      // Update cumulative data
      cumulativeData = if (cumulativeData.isEmpty) chunkDF else cumulativeData.union(chunkDF)

      println(s"Iteration $i/$iterations")
      println(s"Using fixed k = $k")

      // Apply K-Means on the cumulative data
      val kmeans = new KMeans()
        .setK(k)
        .setSeed(1L)
        .setFeaturesCol("features")
        .setPredictionCol("clusterLabel")

      val model = kmeans.fit(cumulativeData)
      predictions = model.transform(cumulativeData)  // Update predictions

      // Print intermediate results
      println(s"\nAfter processing chunk $i/$iterations:")
      val (patterns, nodeIdToClusterLabel) = createPatternsFromKMeansClusters(predictions)
      // Optionally evaluate clustering
      // evaluateClustering(predictions, nodeIdToClusterLabel)
    }

    predictions  // Return predictions instead of cumulativeData
  }

  // Function to create patterns from K-Means clusters
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
      .as[(String, Int)]
      .map { case (nodeId, clusterLabel) =>
        nodeId.toLong -> s"Cluster_$clusterLabel"
      }.collect().toMap

    // Process clustersAggDF to create patterns
    val patterns = clustersAggDF.rdd.map { row =>
      val clusterLabel = row.getAs[Int]("clusterLabel")
      val nodeIds = row.getAs[Seq[String]]("nodeIds")
      val clusterSize = row.getAs[Long]("clusterSize")

      // Find common properties where sum equals clusterSize
      val commonProperties = propertyCols.filter { colName =>
        val colSum = row.getAs[Any](colName) match {
          case n: Number => n.longValue()
          case _ => 0L
        }
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

  // Function to extract features from a pattern for similarity computation
  def extractPatternFeatures(pattern: Pattern): Set[String] = {
    val nodeProperties = pattern.nodes.flatMap(_.properties.keys)
    val nodeLabels = pattern.nodes.map(_.label)
    val edgeRelationshipTypes = pattern.edges.map(_.relationshipType)
    val features = nodeProperties ++ nodeLabels ++ edgeRelationshipTypes

    // Optional: Include property values and edge properties for more detailed features
    // val nodePropertyValues = pattern.nodes.flatMap(_.properties.map { case (k, v) => s"$k=$v" })
    // val edgeProperties = pattern.edges.flatMap(_.properties.map { case (k, v) => s"$k=$v" })
    // val features = nodeProperties ++ nodePropertyValues ++ nodeLabels ++ edgeRelationshipTypes ++ edgeProperties

    features.toSet
  }

  // Function to merge similar patterns based on Jaccard similarity
  def mergeSimilarPatterns(patterns: Array[Pattern], similarityThreshold: Double): Array[Pattern] = {
    val patternFeatures: mutable.ArrayBuffer[(Pattern, Set[String])] = patterns.map { pattern =>
      (pattern, extractPatternFeatures(pattern))
    }.to[mutable.ArrayBuffer]

    val mergedPatterns = mutable.ArrayBuffer[Pattern]()

    while (patternFeatures.nonEmpty) {
      val (currentPattern, currentFeatures) = patternFeatures.remove(0)
      var mergedPattern = currentPattern
      var mergedFeatures = currentFeatures

      var i = 0
      while (i < patternFeatures.length) {
        val (patternB, featuresB) = patternFeatures(i)
        val intersectionSize = mergedFeatures.intersect(featuresB).size.toDouble
        val unionSize = mergedFeatures.union(featuresB).size.toDouble
        val jaccardSimilarity = if (unionSize > 0) intersectionSize / unionSize else 0.0

        println(s"Comparing patterns '${mergedPattern.nodes.head.patternId}' and '${patternB.nodes.head.patternId}': Jaccard Similarity = $jaccardSimilarity")

        if (jaccardSimilarity >= similarityThreshold) {
          // Merge patternB into mergedPattern
          mergedPattern = mergeTwoPatterns(mergedPattern, patternB)
          // Update mergedFeatures
          mergedFeatures = extractPatternFeatures(mergedPattern)
          // Remove patternB from patternFeatures
          patternFeatures.remove(i)
          // Logging
          println(s"Merged pattern '${patternB.nodes.head.patternId}' into pattern '${mergedPattern.nodes.head.patternId}' with similarity $jaccardSimilarity")
          // Reset i to 0 to re-evaluate from the beginning
          i = 0
        } else {
          i += 1
        }
      }

      mergedPatterns += mergedPattern
    }

    mergedPatterns.toArray
  }

  // Function to merge two patterns
  def mergeTwoPatterns(patternA: Pattern, patternB: Pattern): Pattern = {
    // Merge nodes by label
    val mergedNodesMap = (patternA.nodes ++ patternB.nodes).groupBy(_.label).map {
      case (label, nodes) =>
        val mergedProperties = nodes.flatMap(_.properties).toMap
        label -> Node(label, mergedProperties, patternId = nodes.head.patternId)
    }
    val mergedNodes = mergedNodesMap.values.toList

    // Merge edges
    val mergedEdges = (patternA.edges ++ patternB.edges)
      .groupBy(e => (e.startNode.label, e.relationshipType, e.endNode.label))
      .map(_._2.head).toList

    // Merge constraints
    val mergedConstraints = (patternA.constraints ++ patternB.constraints).distinct

    new Pattern(mergedNodes, mergedEdges, mergedConstraints)
  }

  // Function to update node-to-cluster label mapping after merging
  def updateNodeIdToClusterLabel(
    originalNodeIdToClusterLabel: Map[Long, String],
    mergedPatterns: Array[Pattern]
  ): Map[Long, String] = {
    // Map old pattern labels to new merged pattern labels
    val oldToNewLabels = mergedPatterns.flatMap { pattern =>
      val newLabel = pattern.nodes.head.label
      pattern.nodes.map(node => node.label -> newLabel)
    }.toMap

    // Update the nodeIdToClusterLabel mapping
    val updatedMapping = originalNodeIdToClusterLabel.map { case (nodeId, oldLabel) =>
      val newLabel = oldToNewLabels.getOrElse(oldLabel, oldLabel)
      nodeId -> newLabel
    }

    updatedMapping
  }
}
