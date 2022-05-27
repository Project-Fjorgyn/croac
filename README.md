# croac
Counting and Recognition using Omnidirectional Acoustic Capture (CROAC)

## Running the Model
```python
import numpy as np
import seaborn as sns
import pandas as pd

from croak.model import PhasedArrayModel

model = PhasedArrayModel(
    omega=2, M=10, N=1, d_x=1, d_y=1, D=1, P=1
)
theta = model.theta
phi = model.phi
p = model.compute_P()
d = 10*np.log10(p/np.max(p))
sns.lineplot(x=theta[phi == 0], y=d[phi == 0]);
```
