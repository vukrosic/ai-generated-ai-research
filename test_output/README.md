# AI Curve Fitting Research

## Overview

This repository contains a comprehensive AI research project focused on analyzing the effectiveness of different neural network architectures and optimization techniques for polynomial curve fitting tasks.

## Results

### Experiment Summary

- **Total Experiments**: 3
- **Success Rate**: 100.0%
- **Average Training Time**: 35.00 seconds

### Best Performing Models

| Rank | Architecture | Optimizer | Degree | Val Loss | Train Loss | Time (s) |
|------|-------------|-----------|---------|----------|------------|----------|
| 1 | linear | sgd | 2 | 0.002000 | 0.001000 | 20.00 |
| 2 | shallow | adam | 3 | 0.003000 | 0.001500 | 35.00 |
| 3 | deep | rmsprop | 4 | 0.004000 | 0.002000 | 50.00 |


## Key Findings

1. **Best Architecture**: Linear networks achieved the lowest validation loss
2. **Best Optimizer**: SGD demonstrated superior performance
3. **Training Efficiency**: Experiments completed in an average of 35.0 seconds

## Usage

```bash
python main.py --config configs/example_config.json
```

---
*Generated on 2025-08-06 13:33:09*
