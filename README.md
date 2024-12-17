# MITHRIL: Schema Discovery for Property Graphs

## Overview

**MITHRIL** is a tool is designed to discover schemas within **Property Graph Databases**. It supports incremental schmea discovery and helps identify the structure, patterns, and relationships in graph data, facilitating the understanding and exploration of datasets.

The project supports schema discovery on popular datasets like **LDBC**, **FIB25**, and **MB6**, and integrates with **Neo4j** for seamless graph data management.

---

## Table of Contents

1. [Installation](#installation)  
2. [Neo4j Setup](#neo4j-setup)  
3. [Dataset Preparation](#dataset-preparation)  
4. [Usage](#usage)  
   - [Option 1: Using the Automated Script](#option-1-using-the-automated-script)  
   - [Option 2: Manual Execution](#option-2-manual-execution)  
   - [Incremental Script](#incremental-script)  
5. [Post-Import Steps](#post-import-steps)  
6. [Notes](#notes)  
7. [License](#license)  
8. [Contributions](#contributions)

---

## Installation

To install and set up **MITHRIL**, follow these steps:

### 1. Clone the Repository

```
git clone https://github.com/sophisid/MITHRIL.git
cd MITHRIL/
```

### 2. Build the Project

Navigate to the **`schemadiscovery`** directory and build the project using `sbt` (Scala Build Tool):

```
cd schemadiscovery
sbt compile
```

---

## Neo4j Setup

**MITHRIL** relies on Neo4j for managing and querying the property graphs. To set up Neo4j:

### 1. Download and Install Neo4j Community Edition 4.4.0

```
wget https://dist.neo4j.org/neo4j-community-4.4.0-unix.tar.gz
tar -xzf neo4j-community-4.4.0-unix.tar.gz
cd neo4j-community-4.4.0
```

You can set the `NEO4J_DIR` environment variable to this directory:

```
export NEO4J_DIR=$(pwd)
```

### 2. Start the Neo4j Server

```
bin/neo4j start
```

Neo4j will be available at [http://localhost:7474](http://localhost:7474).

### 3. Configure Neo4j

1. Set the **initial password** for the default user `neo4j` when prompted.
2. Access the Neo4j browser and verify the connection.

---

## Dataset Preparation

The project includes **evaluation datasets** (FIB25, LDBC, MB6) that need to be unzipped and loaded into Neo4j.

### 1. Unzip the Datasets

```
cd datasets
unzip FIB25/fib25_neo4j_inputs.zip
unzip LDBC/ldbc_neo4j_inputs.zip
unzip MB6/mb6_neo4j_inputs1.zip
```

### 2. Load Datasets into Neo4j

Before importing, ensure that:

1. Neo4j is stopped if currently running.
2. `NEO4J_DIR` is set to your Neo4j installation directory.
3. `current_dataset_dir` is set to the path containing the dataset CSV files.

**General Preparation Steps:**

```
cd $NEO4J_DIR
bin/neo4j stop   # Stop Neo4j if running
rm -rf $NEO4J_DIR/data/databases/neo4j  # Delete old database if needed

export current_dataset_dir=<path-to-datasets>
```

---

### **LDBC Dataset Import (using '|' delimiter)**

```
$NEO4J_DIR/bin/neo4j-admin import --database=neo4j --delimiter='|' \
    --nodes=Comment="$current_dataset_dir/comment_0_0_corrupted.csv" \
    --nodes=Forum="$current_dataset_dir/forum_0_0_corrupted.csv" \
    --nodes=Person="$current_dataset_dir/person_0_0_corrupted.csv" \
    --nodes=Post="$current_dataset_dir/post_0_0_corrupted.csv" \
    --nodes=Place="$current_dataset_dir/place_0_0_corrupted.csv" \
    --nodes=Organisation="$current_dataset_dir/organisation_0_0_corrupted.csv" \
    --nodes=TagClass="$current_dataset_dir/tagclass_0_0_corrupted.csv" \
    --nodes=Tag="$current_dataset_dir/tag_0_0_corrupted.csv" \
    --relationships=HAS_CREATOR="$current_dataset_dir/comment_hasCreator_person_0_0_corrupted.csv" \
    --relationships=HAS_TAG="$current_dataset_dir/comment_hasTag_tag_0_0_corrupted.csv" \
    --relationships=IS_LOCATED_IN="$current_dataset_dir/comment_isLocatedIn_place_0_0_corrupted.csv" \
    --relationships=REPLY_OF="$current_dataset_dir/comment_replyOf_comment_0_0_corrupted.csv" \
    --relationships=REPLY_OF="$current_dataset_dir/comment_replyOf_post_0_0_corrupted.csv" \
    --relationships=CONTAINER_OF="$current_dataset_dir/forum_containerOf_post_0_0_corrupted.csv" \
    --relationships=HAS_MEMBER="$current_dataset_dir/forum_hasMember_person_0_0_corrupted.csv" \
    --relationships=HAS_MODERATOR="$current_dataset_dir/forum_hasModerator_person_0_0_corrupted.csv" \
    --relationships=HAS_TAG="$current_dataset_dir/forum_hasTag_tag_0_0_corrupted.csv" \
    --relationships=HAS_INTEREST="$current_dataset_dir/person_hasInterest_tag_0_0_corrupted.csv" \
    --relationships=IS_LOCATED_IN="$current_dataset_dir/person_isLocatedIn_place_0_0_corrupted.csv" \
    --relationships=KNOWS="$current_dataset_dir/person_knows_person_0_0_corrupted.csv" \
    --relationships=LIKES="$current_dataset_dir/person_likes_comment_0_0_corrupted.csv" \
    --relationships=LIKES="$current_dataset_dir/person_likes_post_0_0_corrupted.csv" \
    --relationships=STUDIES_AT="$current_dataset_dir/person_studyAt_organisation_0_0_corrupted.csv" \
    --relationships=WORKS_AT="$current_dataset_dir/person_workAt_organisation_0_0_corrupted.csv" \
    --relationships=HAS_CREATOR="$current_dataset_dir/post_hasCreator_person_0_0_corrupted.csv" \
    --relationships=HAS_TAG="$current_dataset_dir/post_hasTag_tag_0_0_corrupted.csv" \
    --relationships=IS_LOCATED_IN="$current_dataset_dir/post_isLocatedIn_place_0_0_corrupted.csv" \
    --relationships=IS_LOCATED_IN="$current_dataset_dir/organisation_isLocatedIn_place_0_0_corrupted.csv" \
    --relationships=IS_PART_OF="$current_dataset_dir/place_isPartOf_place_0_0_corrupted.csv" \
    --relationships=HAS_TYPE="$current_dataset_dir/tag_hasType_tagclass_0_0_corrupted.csv" \
    --relationships=IS_SUBCLASS_OF="$current_dataset_dir/tagclass_isSubclassOf_tagclass_0_0_corrupted.csv"
```

---

### **MB6 Dataset Import (using ',' delimiter)**

```
$NEO4J_DIR/bin/neo4j-admin import --database=neo4j --delimiter=',' \
    --nodes=Meta="$current_dataset_dir/Neuprint_Meta_mb6_corrupted.csv" \
    --nodes=Neuron="$current_dataset_dir/Neuprint_Neurons_mb6_corrupted.csv" \
    --relationships=CONNECTS_TO="$current_dataset_dir/Neuprint_Neuron_Connections_mb6_corrupted.csv" \
    --nodes=SynapseSet="$current_dataset_dir/Neuprint_SynapseSet_mb6_corrupted.csv" \
    --relationships=CONNECTS_TO="$current_dataset_dir/Neuprint_SynapseSet_to_SynapseSet_mb6_corrupted.csv" \
    --relationships=CONTAINS="$current_dataset_dir/Neuprint_Neuron_to_SynapseSet_mb6_corrupted.csv" \
    --nodes=Synapse="$current_dataset_dir/Neuprint_Synapses_mb6_corrupted.csv" \
    --relationships=SYNAPSES_TO="$current_dataset_dir/Neuprint_Synapse_Connections_mb6_corrupted.csv" \
    --relationships=CONTAINS="$current_dataset_dir/Neuprint_SynapseSet_to_Synapses_mb6_corrupted.csv"
```

---

### **FIB25 Dataset Import (using ',' delimiter)**

```
$NEO4J_DIR/bin/neo4j-admin import --database=neo4j --delimiter=',' \
    --nodes=Meta="$current_dataset_dir/Neuprint_Meta_fib25_corrupted.csv" \
    --nodes=Neuron="$current_dataset_dir/Neuprint_Neurons_fib25_corrupted.csv" \
    --relationships=CONNECTS_TO="$current_dataset_dir/Neuprint_Neuron_Connections_fib25_corrupted.csv" \
    --nodes=SynapseSet="$current_dataset_dir/Neuprint_SynapseSet_fib25_corrupted.csv" \
    --relationships=CONNECTS_TO="$current_dataset_dir/Neuprint_SynapseSet_to_SynapseSet_fib25_corrupted.csv" \
    --relationships=CONTAINS="$current_dataset_dir/Neuprint_Neuron_to_SynapseSet_fib25_corrupted.csv" \
    --nodes=Synapse="$current_dataset_dir/Neuprint_Synapses_fib25_corrupted.csv" \
    --relationships=SYNAPSES_TO="$current_dataset_dir/Neuprint_Synapse_Connections_fib25_corrupted.csv" \
    --relationships=CONTAINS="$current_dataset_dir/Neuprint_SynapseSet_to_Synapses_fib25_corrupted.csv"
```

---

## Post-Import Steps

1. **Verify the Import**:
   ```
   $NEO4J_DIR/bin/neo4j-admin check-consistency --database=neo4j
   ```

2. **Start the Neo4j Server**:
   ```
   cd $NEO4J_DIR
   bin/neo4j start
   ```
   
3. **Access the Neo4j Browser**:  
   Open [http://localhost:7474](http://localhost:7474) to visualize the graph and run queries.

---

## Notes

1. **Delimiters**:  
   - Use `|` for **LDBC** dataset.  
   - Use `,` for **MB6** and **FIB25** datasets.

2. **CSV Format Requirements**:  
   - Nodes must have a unique `id` field.
   - Relationships must have `:START_ID`, `:END_ID`, and `:TYPE` columns.
   
3. **Reloading Data**:  
   If you need to re-import, remove the database first:
   ```
   rm -rf $NEO4J_DIR/data/databases/neo4j
   ```

---

## Usage

Once the setup is complete and the datasets are loaded, you can run **MITHRIL** to perform schema discovery.

**MITHRIL** can be executed in multiple ways:

### Option 1: Using the Automated Script

Scripts (e.g., `run_mithril_ldbc.sh`, `run_mithril_fib25.sh`, `run_mithril_mb6.sh`) are provided to automate the entire process:

1. Set environment variables and directories inside these scripts.
2. Make them executable:
   ```
   chmod +x run_mithril_ldbc.sh
   chmod +x run_mithril_fib25.sh
   chmod +x run_mithril_mb6.sh
   ```
3. Run the script for the desired dataset:
   ```
   ./run_mithril_ldbc.sh
   ./run_mithril_fib25.sh
   ./run_mithril_mb6.sh
   ```

The scripts handle:
- Removing and re-extracting Neo4j.
- Importing data.
- Running schema discovery (LSH clustering).
- Stopping Neo4j and cleaning up.

### Option 2: Manual Execution

If you prefer more control, follow the manual steps:
1. Stop Neo4j, remove old database, re-extract Neo4j.
2. Import the desired dataset with `neo4j-admin import`.
3. Start Neo4j.
4. Run:
   ```
   cd schemadiscovery
   sbt "run l"  # LSH clustering
   ```
5. Stop Neo4j and clean up if needed.

### Incremental Script

An incremental script is also provided (an example shown below) to process datasets incrementally. This script follows a similar pattern but runs the Scala program in an "incremental" mode (indicated by `sbt "run l i 500000"`) after importing the dataset. It then stops Neo4j, cleans up processes, and moves on to the next dataset.

**Usage Incremental Script:**  
1. Set environment variables and directories inside these scripts.
2. Make them executable:
   ```
   chmod +x run_mithril_ldbc_incremental.sh
   ```
3. Run the script for the desired dataset:
   ```
   ./run_mithril_ldbc_incremental.sh
   ```
---

## License

This project is licensed under the **MIT License**. See the [LICENSE](LICENSE) file for details.

---

## Contributions

Contributions to **MITHRIL** are welcome! If you find bugs, have suggestions, or want to contribute features, feel free to open an issue or submit a pull request.

For questions, feedback, or support, please contact the repository maintainer.