import org.apache.spark.sql.{DataFrame, SparkSession}
import org.apache.spark.ml.feature.VectorAssembler
import org.apache.spark.sql.functions._
import org.apache.spark.sql.types._
import utils.DataTypeMapping

object DataProcessor {

  // Function to create binary matrix of properties
  def createBinaryMatrix(df: DataFrame): (DataFrame, Map[String, DataType]) = {
    val spark = df.sparkSession
    import spark.implicits._

    // Get all property columns except node ID
    val propertyCols = df.columns.filterNot(_ == "_nodeId")

    // Create binary columns indicating presence of each property
    val binaryDF = propertyCols.foldLeft(df) { (tempDF, colName) =>
      tempDF.withColumn(colName, when(col(colName).isNotNull, 1).otherwise(0))
    }

    // Get property data types
    val propertyDataTypes = df.schema.fields
      .filter(f => propertyCols.contains(f.name))
      .map(f => f.name -> f.dataType)
      .toMap

    // Map data types to type codes
    val propertyTypeCodes: Map[String, Int] = propertyDataTypes.map { case (propName, dataType) =>
      val typeCode = DataTypeMapping.dataTypeToCode.getOrElse(dataType, -1)
      propName -> typeCode
    }

    // Add type code features
    var dfWithTypeCodes = binaryDF

    propertyCols.foreach { colName =>
      val typeCode = propertyTypeCodes(colName)
      dfWithTypeCodes = dfWithTypeCodes.withColumn(s"${colName}_typeCode", lit(typeCode))
    }

    println(s"Binary matrix created with ${dfWithTypeCodes.columns.length} properties.")
    println("Sample data from binary matrix:")
    dfWithTypeCodes.show(5)

    // Επιστρέψτε το DataFrame και το propertyDataTypes
    (dfWithTypeCodes, propertyDataTypes)
  }

  // Function to assemble features for K-Means
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

  // Function to get property data types
  def getPropertyDataTypes(df: DataFrame): Map[String, DataType] = {
    val propertyCols = df.columns.filterNot(_ == "_nodeId")
    val propertyDataTypes = df.schema.fields
      .filter(f => propertyCols.contains(f.name))
      .map(f => f.name -> f.dataType)
      .toMap
    propertyDataTypes
  }
}
