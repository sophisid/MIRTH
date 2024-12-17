#!/bin/bash

ROOT_DIR="your_root/directory"
PROJECT_DIR="your_project/directory"
DATASETS_DIR="$PROJECT_DIR/noisy_datasets/MB6"
SCRIPT_DIR="$PROJECT_DIR/scripts"
SCHEMA_DISCOVERY_DIR="$PROJECT_DIR/schemadiscovery"
NEO4J_TAR="neo4j-community.tar.gz"
NEO4J_VERSION="neo4j-community-4.4.0"
NEO4J_DIR="$NEO4J_VERSION"
OUTPUT_BASE_DIR="$PROJECT_DIR/output"
NEO4J_PORT=7687  # Specify the Neo4j port to kill the process dynamically

mkdir -p "$OUTPUT_BASE_DIR"

datasets=($(ls "$DATASETS_DIR" | grep "corrupted"))

# Loop through each dataset
for dataset in "${datasets[@]}"
do
    echo "==============================="
    echo "Processing Dataset: $dataset"
    echo "==============================="

    # Current dataset directory
    current_dataset_dir="$DATASETS_DIR/$dataset"

    # Step 1: Remove the current Neo4j database directory
    echo "Removing $NEO4J_VERSION directory..."
    rm -rf "$NEO4J_DIR"

    # Step 2: Extract Neo4j from the tar.gz
    echo "Extracting Neo4j from $NEO4J_TAR..."
    tar -xzvf "$NEO4J_TAR"

    # Step 3: Import the CSV files into Neo4j
    echo "Importing data into Neo4j..."
    "$NEO4J_DIR/bin/neo4j-admin" import --database=neo4j --delimiter='|' \
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

    # Step 4: Start Neo4j
    echo "Starting Neo4j..."
    "$NEO4J_DIR/bin/neo4j" start

    # Wait for Neo4j to start
    echo "Waiting for Neo4j to start..."
    sleep 100  # Adjust the time if necessary

    # Step 5: Run your Scala program with LSH clustering
    echo "Running Scala program with LSH clustering incremental..."
    cd "$SCHEMA_DISCOVERY_DIR" || { echo "Couldn't change directory to $SCHEMA_DISCOVERY_DIR"; exit 1; }
    sbt "run l i 500000" > "$OUTPUT_BASE_DIR/output_LSH_LDBC_incremental_${dataset#corrupted}.txt"

    # Step 6: Run your Scala program with K-Means clustering
    # echo "Running Scala program with K-Means clustering incremental..."
    # sbt "run k i 500000" > "$OUTPUT_BASE_DIR/output_KMeans_LDBC_incremental_${dataset#corrupted}.txt"
    cd "$ROOT_DIR" || { echo "Couldn't change back to $ROOT_DIR"; exit 1; }

    # Step 7: Stop Neo4j
    echo "Stopping Neo4j..."
    "$NEO4J_DIR/bin/neo4j" stop
    sleep 100  # Wait 60 seconds to ensure Neo4j has stopped

    # Step 8: Dynamically kill any process on port 7687
    echo "Checking and killing any process on port $NEO4J_PORT..."
    PID=$(lsof -t -i :$NEO4J_PORT)

    if [ -z "$PID" ]; then
        echo "No process found on port $NEO4J_PORT."
    else
        echo "Killing process with PID: $PID"
        kill -9 $PID
        if [ $? -eq 0 ]; then
            echo "Process $PID successfully killed."
        else
            echo "Failed to kill process $PID."
        fi
    fi

    echo "Finished processing dataset: $dataset"
    echo ""
done

echo "All datasets have been processed."
