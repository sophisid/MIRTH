import org.apache.spark.sql.{DataFrame, SparkSession, Row}
import org.apache.spark.ml.feature._
import org.apache.spark.ml.linalg.{Vector, Vectors}
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import scala.collection.mutable
import org.apache.spark.broadcast.Broadcast
import org.apache.spark.ml.feature.MinHashLSH
import scala.math._

object Clustering {
  // Function to calculate the ideal number of hash tables
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

    println(s"Calculated numHashTables: $numHashTablesCalculated, Using numHashTables: $numHashTables")


    // Apply MinHash LSH
    val mh = new MinHashLSH()
      .setNumHashTables(numHashTables)
      .setInputCol("features")
      .setOutputCol("hashes")

    val model = mh.fit(featureDF)
    val lshDF = model.transform(featureDF)

    lshDF
  }

  // Function to create patterns from clusters
  def createPatternsFromClusters(df: DataFrame): (Array[Pattern], Map[Long, String]) = {
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
        properties = commonProperties.map(prop => prop -> 1).toMap
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
        val startNode = Node(label = startLabel, properties = Map.empty)
        val endNode = Node(label = endLabel, properties = Map.empty)

        Edge(
          startNode = startNode,
          relationshipType = relationshipType,
          endNode = endNode,
          properties = properties
        )
      }
    }.collect()

    edges
  }
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
  
}
