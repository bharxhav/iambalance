# iambalance

iambalance is a Python package for advanced oversampling of imbalanced datasets. It combines multiple oversampling techniques, including SMOTE and ADASYN, with a unique purity measure to ensure high-quality synthetic samples.

## Features

- Combine multiple oversampling techniques in a single interface
- Customizable oversampling strategies
- Purity measure to assess oversampling quality
- Visualization tools for oversampling process and results

## Installation

You can install iambalance using pip:

```bash
pip install iambalance
```

## Quick Start

Here's a simple example of how to use iambalance:

```python
from iambalance import Oversampler
import pandas as pd

# Load your imbalanced dataset
X = pd.read_csv('your_features.csv')
y = pd.read_csv('your_target.csv')

# Initialize the Oversampler
oversampler = Oversampler(methods=['smote', 'adasyn'], oversample_size=[0.7, 0.3])

# Fit and resample your data
X_resampled, y_resampled = oversampler.fit_resample(X, y)

# Get purity trend
purity_trend = oversampler.get_purity_trend()

# Plot purity trend
oversampler.plot_purity_trend()
```
