#!/bin/bash
wait_for_running() {
    local job_id=$1
    echo "Waiting for HQ server with jobid ${job_id} to start running..."

    while true; do
        # Get job state
        state=$(squeue -j "$job_id" -h -o %t)

        # Check if job exists and is running
        if [ "$state" = "R" ]; then
            echo "HQ server ${job_id} is now running"
            return 0
        elif [ -z "$state" ]; then
            echo "HQ server ${job_id} is not in queue anymore - may have failed"
            return 1
        fi

        echo "HQ server ${job_id} is in state: ${state}"
        sleep 10
    done
}

# Number of workers
if [ -z "$1" ]; then
    echo "Usage: $0 <NUM_WORKERS> <CPUS_PER_WORKER>"
    exit 1
fi
NUM_WORKERS=$1

# Number of CPUs per worker
if [ -z "$2" ]; then
    echo "Usage: $0 <NUM_WORKERS> <CPUS_PER_WORKER>"
    exit 1
fi
CPUS_PER_WORKER=$2

SCRIPT_DIR=$(dirname "$(realpath "$0")")

echo "Starting HQ server"

job_id=$(sbatch --parsable $SCRIPT_DIR/start_hq_server.slurm)

if [ $? -ne 0 ]; then
    echo "Error submitting HQ server"
    exit 1
fi

echo "Submitted HQ server with jobid $job_id"

# Wait for job to start running
wait_for_running "$job_id"

echo "HQ server is running"

echo "Starting $NUM_WORKERS HQ workers with $CPUS_PER_WORKER CPUs each"

# Start workers
job_ids=()
for i in $(seq 1 "$NUM_WORKERS"); do
    echo "Starting worker $i"
    job_id=$(sbatch --parsable $SCRIPT_DIR/start_hq_worker.slurm $CPUS_PER_WORKER)
    if [ $? -ne 0 ]; then
        echo "Error submitting worker $i"
        exit 1
    fi
    job_ids+=("$job_id")
done
