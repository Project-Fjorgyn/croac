# croac
Counting and Recognition using Omnidirectional Acoustic Capture (CROAC)

## Running the Model
```python
import numpy as np
import seaborn as sns
import pandas as pd

from croac.model import PhasedArrayModel

# run the model
model = PhasedArrayModel(
    omega=2, M=10, N=1, d_x=1, d_y=1, D=1, S=1
)
p = model.compute_P()
d = 10*np.log10(p/np.max(p))
sns.lineplot(x=model.theta, y=d);

# get a distribution to fit from the model
y = model.P
X = np.array([model.theta, model.phi]).T

# fit 
model = PhasedArrayModel(
    omega=2, M=5, N=1, d_x=1, d_y=1, D=1, S=2
)
model.fit(X,y)
sns.lineplot(x=model.theta, y=model.P);
sns.lineplot(x=model.theta, y=model.O);
```
