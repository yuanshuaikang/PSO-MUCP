# PSO-MUCP Algorithm

This document provides detailed information about the PSO-MUCP algorithm and its comparison algorithms (EPA, EHUCM, and CQ-HUCPM) for mining High Utility Co-location Patterns (HUCP) from spatial datasets.

## Table of Contents
1. [Overview](#overview)
2. [Algorithms](#algorithms)
3. [Parameters](#parameters)
4. [Input Data Format](#input-data-format)
5. [Output](#output)
6. [Usage](#usage)
7. [Performance Metrics](#performance-metrics)
8. [References](#references)

## Overview

PSO-MUCP (Pruning Strategies Optimized Mixed-Utility Co-location Pattern Mining Algorithm) is an advanced algorithm for discovering spatial co-location patterns that are both prevalent and valuable. It incorporates utility values (both positive and negative) into the pattern discovery process and introduces four targeted pruning strategies to improve efficiency.

## Algorithms

The repository contains implementations of four algorithms:

1. **PSO-MUCP (main.py)** - The proposed algorithm supporting both positive and negative utilities
2. **EPA (EPA1.3.py)** - Extended Pruning Algorithm (baseline)
3. **EHUCM (EHUCM1.2.py)** - Efficient High Utility Co-location Miner (baseline)
4. **CQ-HUCPM (CQ-HUCPM1.1.py)** - Clique Query based High Utility Co-location Pattern Miner (baseline)

## Parameters

| Parameter | Type | Description | Typical Value Range |
|-----------|------|-------------|---------------------|
| `D` | float | Distance threshold for determining spatial neighborhood relationships. Larger values generate more neighbor relations. | 500-2000 (units depend on dataset) |
| `Min_utility` | float | Minimum utility threshold (0-1). Patterns with utility ratio below this will be filtered. | 0.1-0.5 |
| `Utility` | dict | Dictionary mapping features to their utility values (can be positive or negative). Example: `{'A': 2, 'B': -1}` | Varies by application |
| `Instance` | list | List of spatial instances, each containing [feature+ID, x-coord, y-coord] | Loaded from input file |
| `start_time` | float | Timestamp for tracking execution time | Set automatically |

## Input Data Format

The algorithms expect CSV files with the following format:
```
Feature,Instance,LocX,LocY,Checkin
A,1,123.45,678.90,1
B,1,124.50,677.85,1
...
```
ps: For ease of processing, the corresponding features are replaced with letters, for example, {restaurant} is replaced with {A}.

Columns:
1. Feature: The spatial feature type (e.g., 'A', 'B')
2. Instance: Instance identifier
3. LocX: X-coordinate of the instance
4. LocY: Y-coordinate of the instance
5. Checkin: (Optional) Additional attribute

## Output

Each algorithm outputs:
1. High utility co-location patterns (patterns with utility ≥ min_utility)
2. Number of high utility patterns found
3. Execution time
4. Memory usage (in MB)

## Usage

1. Prepare your input CSV file
2. Set parameters in the script:
   ```python
   Utility = {'A': 2, 'B': 4, 'C': 8}  # Feature utilities
   Min_utility = 0.5  # Minimum utility threshold
   D = 1300  # Distance threshold
   ```
3. Update the file path:
   ```python
   f = open(r"path/to/your/data.csv", "r", encoding="UTF-8")
   ```
4. Run the script:
   ```bash
   python main.py  # For PSO-MUCP
   ```

## Performance Metrics

The algorithms measure and report:
1. **Execution time**: Total runtime in seconds
2. **Memory usage**: Peak memory consumption in MB
3. **Pruning ratio**: Percentage of candidate patterns eliminated by pruning strategies
4. **Pattern completeness**: Ensures all true high utility patterns are found

## References

For detailed algorithm descriptions and experimental results, please refer to:
- The included paper "Efficient_Pruning_Strategies_for_Mining_High_Utility_Co-location_Patterns_with_Negative_Utility_Features.pdf"
- Original papers for baseline algorithms (EPA, EHUCM, CQ-HUCPM)

For any questions or issues, please contact the authors listed in the paper.
