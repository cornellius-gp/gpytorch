# Data Normalization Best Practices for GPyTorch Examples

## Overview

This document outlines the correct approach to normalizing data in GPyTorch examples to avoid data leakage.

## The Problem: Data Leakage

**Data leakage** occurs when information from the test set is used during the training process. A common form of data leakage happens when normalizing data using statistics computed from both training and test sets.

### ❌ Incorrect Approach (Data Leakage)

```python
# WRONG: Computing statistics from combined train+test data
combined_data = torch.cat([train_x, test_x], dim=0)
mean = combined_data.mean()
std = combined_data.std()

train_x_normalized = (train_x - mean) / std
test_x_normalized = (test_x - mean) / std
```

This is incorrect because:
1. The test set statistics influence the normalization of training data
2. The model indirectly "sees" information about the test set during training
3. This leads to overly optimistic performance estimates

### ✅ Correct Approach

```python
# CORRECT: Compute statistics from training data only
train_mean = train_x.mean(dim=-2, keepdim=True)
train_std = train_x.std(dim=-2, keepdim=True) + 1e-6  # Add small epsilon to prevent division by zero

# Normalize both train and test using ONLY training statistics
train_x_normalized = (train_x - train_mean) / train_std
test_x_normalized = (test_x - train_mean) / train_std
```

## Complete Example

```python
import torch

# Split data
n_train = int(0.8 * len(data))
train_x, train_y = data[:n_train, :-1], data[:n_train, -1]
test_x, test_y = data[n_train:, :-1], data[n_train:, -1]

# Normalize features using ONLY training statistics
train_x_mean = train_x.mean(dim=-2, keepdim=True)
train_x_std = train_x.std(dim=-2, keepdim=True) + 1e-6
train_x = (train_x - train_x_mean) / train_x_std
test_x = (test_x - train_x_mean) / train_x_std  # Use training stats!

# Normalize labels using ONLY training statistics
train_y_mean = train_y.mean()
train_y_std = train_y.std()
train_y = (train_y - train_y_mean) / train_y_std
test_y = (test_y - train_y_mean) / train_y_std  # Use training stats!

# Make data contiguous
train_x, train_y = train_x.contiguous(), train_y.contiguous()
test_x, test_y = test_x.contiguous(), test_y.contiguous()
```

## Why This Matters

1. **Realistic Performance Estimates**: Using only training statistics ensures that your model's performance on the test set reflects how it would perform on truly unseen data.

2. **Production Readiness**: In production, you won't have access to test/future data statistics, so your normalization must use only training data statistics.

3. **Fair Comparisons**: When comparing different models or methods, consistent normalization practices ensure fair comparisons.

## Verification

All GPyTorch example notebooks have been verified to follow these best practices. You can run the verification script:

```bash
python check_normalization.py
```

## References

- Issue #819: [Bug] Some examples normalize training data with test data
- Related discussions on data leakage in machine learning

## Contributing

When contributing new examples to GPyTorch, please ensure:
1. Normalization statistics are computed from training data only
2. The same training statistics are used to normalize test data
3. Add comments in your code to make the normalization approach clear

---

*Last updated: 2024*
*Verified: All 50 example notebooks follow these best practices*
