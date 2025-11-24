# Scoring Metrics and Weights

The system evaluates miners using four key metrics, each with an assigned weight:

1. **Compute Units (55%)**: Measures the total computational work performed, calculated as the sum of:
   - Flat sum of bounties (as compute units)
   - Compute time
     - Normalized using median performance (tokens-per-second and/or steps-per-second across miners)
     - Multiplied by compute multiplier (based on number and type of GPUs)
   - Using appropriate time measurement methods (step-based, token-based, or raw execution time)
2. **Invocation Count (25%)**: The total number of successful invocations (compute jobs) handled
3. **Unique Chute Score (15%)**: Average number of unique chutes that a miner runs simultaneously, weighted by GPU requirements
4. **Bounty Count (5%)**: The number of bounties received (not the value, just the count)

## Scoring Process

The scoring algorithm follows these steps:

### 1. Data Collection

Queries the database for raw metrics using SQL queries within a specified scoring interval (default: 7 days):

- **Compute metrics**: Uses median computation rates (step time and token time) calculated over the last 2 days to normalize compute units
- **Unique chute metrics**: Calculates GPU-weighted chute counts using the latest GPU count from chute history, with hourly snapshots over the scoring period

### 2. Normalization Process

The system applies different normalization strategies for each metric:

**Standard Metrics (compute_units, invocation_count, bounty_count)**:

- Normalized by dividing each miner's value by the total sum across all miners

**Unique Chute Score**:

- Uses a sophisticated two-tier normalization system:
  - **Above median**: Miners with chute counts `≥` median are normalized using exponent 1.3: `(count / highest_count)^1.3`
  - **Below median**: Miners with chute counts `<` median are normalized using exponent 2.2: `(count / highest_count)^2.2` - After initial normalization, all unique chute scores are re-normalized to sum to 1.0

### 3. Multi-UID Punishment

Penalizes miners who run multiple nodes with the same coldkey (identity):

- Ranks all miners by their preliminary scores (highest first)
- For each coldkey, only the highest-scoring hotkey receives rewards
- All other hotkeys sharing the same coldkey receive zero score

## GPU-Weighted Chute Calculation

The unique chute score uses a sophisticated GPU-weighting system:

1. **Historical GPU Tracking**: Uses the latest GPU count from `chute_history` for each chute
2. **Hourly Snapshots**: Takes hourly snapshots of active chutes over the scoring period
3. **GPU Weighting**: Each chute contributes its GPU count (defaults to 1 if no history exists) to the miner's score
4. **Time Averaging**: Averages GPU-weighted chute counts across all time points in the scoring period

## Anti-Gaming Mechanisms

The code includes several safeguards against gaming the system:

1. **Multi-UID Punishment**: Prevents miners from gaining advantage by running multiple nodes with the same coldkey
2. **Median Computation Rates**: Uses median values for step/token times calculated over 2 days to resist manipulation
3. **Error Filtering**: Only counts successful invocations (no errors, completed successfully)
4. **Report Filtering**: Excludes invocations that have been reported for issues
5. **GPU History Validation**: Uses historical GPU counts from chute history to prevent gaming through GPU count manipulation
6. **Successful Instance Filtering**: Only considers instances that have had at least one successful invocation in their lifetime
7. **Two-Tier Chute Normalization**: The unique chute score's dual-exponent system (1.3 vs 2.2) rewards miners who maintain above-median chute diversity while penalizing those below median

This scoring system aims to fairly distribute rewards based on actual computational work performed, with mechanisms to prevent gaming and ensure network health.
