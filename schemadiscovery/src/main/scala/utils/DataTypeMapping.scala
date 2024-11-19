package utils

import org.apache.spark.sql.types._

object DataTypeMapping {
  val dataTypeToCode: Map[DataType, Int] = Map(
    StringType -> 1,
    IntegerType -> 2,
    FloatType -> 3,
    DoubleType -> 4,
    DateType -> 5,
    BooleanType -> 6,
    LongType -> 7,
    ShortType -> 8,
    ByteType -> 9
  )
}
