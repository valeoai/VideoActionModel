#!/bin/bash

# Assert that $1 and $2 are defined
if [ -z "$1" ] || [ -z "$2" ]; then
    echo "Usage: $0 <INPUT_DIR> <OUTFILE>"
    exit 1
fi

INPUT_DIR=$1
OUTFILE=$2

find $INPUT_DIR -type f -name "*.jpg" | sort > $OUTFILE
