#!/usr/bin/env bash

# Exit if any command fails
set -e

# Navigate to the "src" directory
cd src

echo "Generating main figures..."
entire_runtime=0
# Iterate notebooks from 1 to 6
for i in {1..6}; do
    echo "Running Figure${i}.ipynb..."

    # Record the start time (in seconds)
    start_time=$(date +%s)

    # Run the notebook without keeping the outputs in the final file
    jupyter nbconvert \
      --to notebook \
      --execute Figure${i}.ipynb \
      --output Figure${i}_executed.ipynb \
      --ExecutePreprocessor.timeout=-1 \
      --ClearOutputPreprocessor.enabled=True

    # Record the end time (in seconds) and compute duration
    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # Convert runtime to minutes (integer approximation)
    runtime_secs=$((runtime))
    entire_runtime=$((entire_runtime + runtime_secs))

    # Print summary
    echo "Figures for Figure ${i} are saved under ../results/Figure${i} and it took ${runtime_secs} seconds."
    rm Figure${i}_executed.ipynb
    echo "--------------------------------------------------------"
done

# Print total runtime
echo "Generating all main figures took ${entire_runtime} seconds."


echo "Generating extended data figures..."
entire_runtime=0
# Iterate notebooks from 1 to 6
for i in {1..12}; do
    # Skip 5
    if [ $i -eq 5 ]; then
        continue
    fi

    echo "Running EDFigure${i}.ipynb..."

    # Record the start time (in seconds)
    start_time=$(date +%s)

    # Run the notebook without keeping the outputs in the final file
    jupyter nbconvert \
      --to notebook \
      --execute EDFigure${i}.ipynb \
      --output EDFigure${i}_executed.ipynb \
      --ExecutePreprocessor.timeout=-1 \
      --ClearOutputPreprocessor.enabled=True

    # Record the end time (in seconds) and compute duration
    end_time=$(date +%s)
    runtime=$((end_time - start_time))

    # Convert runtime to minutes (integer approximation)
    runtime_secs=$((runtime))
    entire_runtime=$((entire_runtime + runtime_secs))

    # Print summary
    echo "Figures for EDFigure ${i} are saved under ../results/EDFigure${i} and it took ${runtime_secs} seconds."
    rm EDFigure${i}_executed.ipynb
    echo "--------------------------------------------------------"
done

# Print total runtime
echo "Generating all extended data figures took ${entire_runtime} seconds."

exit 0