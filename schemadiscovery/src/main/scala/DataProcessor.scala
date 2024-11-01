import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._

object DataProcessor {

  // Function to create binary matrix of properties
  def createBinaryMatrix(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    // Get all property columns except node ID
    val propertyCols = df.columns.filterNot(_ == "_nodeId")

    // Create binary columns indicating presence of each property
    val binaryDF = propertyCols.foldLeft(df) { (tempDF, colName) =>
      tempDF.withColumn(colName, when(col(colName).isNotNull, 1).otherwise(0))
    }
    println(s"Binary matrix created with ${binaryDF.columns.length} properties.")
    println("Sample data from binary matrix:")
    binaryDF.show(5)

    binaryDF
  }

  def assembleFeaturesForKMeans(df: DataFrame): DataFrame = {
    val spark = df.sparkSession
    import spark.implicits._

    // Get all property columns except node ID
    val propertyCols = df.columns.filterNot(_ == "_nodeId")

    // Ensure features are in double format for k-means
    val numericDF = propertyCols.foldLeft(df) { (tempDF, colName) =>
      tempDF.withColumn(colName, when(col(colName).isNotNull, 1.0).otherwise(0.0))
    }

    // Assemble features into a vector
    val assembler = new VectorAssembler()
      .setInputCols(propertyCols)
      .setOutputCol("features")

    val featureDF = assembler.transform(numericDF)
    println("Features assembled for k-means clustering:")
    featureDF.select("_nodeId", "features").show(5)

    featureDF
  }
}
