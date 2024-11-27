import org.apache.spark.sql.{DataFrame, Row, SparkSession}
import org.apache.spark.sql.types._
import org.neo4j.driver.{AuthTokens, GraphDatabase}
import scala.collection.JavaConverters._

object DataLoader {

  // Function to load all nodes without labels
  def loadAllNodes(spark: SparkSession): DataFrame = {
    import spark.implicits._

    val uri = "bolt://localhost:7687"
    val user = "neo4j"
    val password = "password"

    val driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))
    val session = driver.session()

    println("Loading all nodes from Neo4j")

    // The query to return property labels
    val result = session.run("MATCH (n) RETURN n, labels(n) AS labels")
    val nodes = result.list().asScala.map { record =>
      val node = record.get("n").asNode()
      val labels = record.get("labels").asList().asScala.map(_.toString)
      val props = node.asMap().asScala.toMap
      // Include node ID and labels
      // Store _nodeId as Long
      props + ("_nodeId" -> node.id()) + ("_labels" -> labels.mkString(":"))
    }

    session.close()
    driver.close()

    // Collect all unique keys to define the schema
    val allKeys = nodes.flatMap(_.keys).toSet

    // Define the schema based on the keys
    val fields = allKeys.map { key =>
      if (key == "_nodeId") {
        StructField(key, LongType, nullable = false)
      } else {
        StructField(key, StringType, nullable = true)
      }
    }.toArray
    val schema = StructType(fields)

    // Convert list of Maps to DataFrame
    val rows = nodes.map { nodeMap =>
      val values = schema.fields.map { field =>
        Option(nodeMap.getOrElse(field.name, null)).map { value =>
          if (field.name == "_nodeId") {
            value.asInstanceOf[Long]
          } else {
            value.toString
          }
        }.orNull
      }
      Row(values: _*)
    }

    val nodesDF = spark.createDataFrame(spark.sparkContext.parallelize(rows.toSeq), schema)
    println(s"Total nodes loaded: ${nodesDF.count()}")
    println("Schema of nodesDF:")
    nodesDF.printSchema()
    nodesDF
  }

  def loadAllRelationships(spark: SparkSession): DataFrame = {
    import spark.implicits._

    val uri = "bolt://localhost:7687"
    val user = "neo4j"
    val password = "password"

    val driver = GraphDatabase.driver(uri, AuthTokens.basic(user, password))
    val session = driver.session()

    println("Loading all relationships from Neo4j")

    val result = session.run("MATCH (n)-[r]->(m) RETURN id(n) AS srcId, id(m) AS dstId, type(r) AS relationshipType, properties(r) AS properties")

    val relationships = result.list().asScala.map { record =>
      val srcId = record.get("srcId").asLong()
      val dstId = record.get("dstId").asLong()
      val relationshipType = record.get("relationshipType").asString()
      val properties = record.get("properties").asMap().asScala.toMap.mapValues(_.toString)

      (srcId, dstId, relationshipType, properties)
    }

    session.close()
    driver.close()

    // DataFrame
    val relationshipsDF = relationships.toDF("srcId", "dstId", "relationshipType", "properties")
    println(s"Total relationships loaded: ${relationshipsDF.count()}")
    relationshipsDF
  }
}
