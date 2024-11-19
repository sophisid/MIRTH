// Define the Node class
case class Node(
  var label: String,
  properties: Map[String, Any],
  propertyTypeCodes: Map[String, Int],
  // isOptional: Boolean = false,
  // minCardinality: Int = 1,
  // maxCardinality: Int = 1,
  patternId: String
)

// Define the Edge class
@SerialVersionUID(1L)
case class Edge(
  startNode: Node,
  relationshipType: String,
  endNode: Node,
  properties: Map[String, Any],
  // isOptional: Boolean = false,
  // minCardinality: Int = 1,
  // maxCardinality: Int = 1,
  patternId: String = ""
) extends Serializable

// Define the Constraint class
case class Constraint(
  field: String,
  operation: String,
  value: Any,
  patternId: String = ""
)

// Define the Pattern class that holds nodes, edges, and constraints
@SerialVersionUID(1L)
class Pattern(
  var nodes: List[Node] = List(),
  var edges: List[Edge] = List(),
  var constraints: List[Constraint] = List()
)extends Serializable 
 {

  // Add a node to the pattern
  def addNode(node: Node): Unit = {
    nodes = nodes :+ node
  }

  // Add an edge to the pattern
  def addEdge(edge: Edge): Unit = {
    if (!edges.exists(e => 
          e.relationshipType == edge.relationshipType && 
          e.startNode.label == edge.startNode.label && 
          e.endNode.label == edge.endNode.label)) {
      edges = edges :+ edge
    }
  }


  // Add a constraint to the pattern
  // def addConstraint(constraint: Constraint): Unit = {
  //   constraints = constraints :+ constraint
  // }

  // Display the pattern including nodes, edges, and constraints
override def toString: String = {
  val nodeStr = nodes.map { node =>
    val propsWithTypes = node.properties.keys.map { prop =>
      val typeCode = node.propertyTypeCodes.getOrElse(prop, -1)
      s"$prop: TypeCode($typeCode)"
    }.mkString("{", ", ", "}")
    s"Node(label=${node.label}, properties=$propsWithTypes)"
  }.mkString(", ")

  val edgeStr = edges.map { edge =>
    s"Edge(relationshipType=${edge.relationshipType}, start=${edge.startNode.label}, end=${edge.endNode.label})"
  }.mkString(", ")

  s"Nodes: [$nodeStr]\nEdges: [$edgeStr]\nConstraints: ${constraints.mkString(", ")}"
}


}
