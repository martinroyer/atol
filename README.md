# atol: python3 package for _ATOL: Measure Vectorisation for Automatic Topologically-Oriented Learning_

**Author**: Martin Royer
**Copyright**: INRIA

# Description

`atol` is a python package implementing an automatic measure vectorisation procedure for topological learning problems. It is based on the ATOL paper https://hal.archives-ouvertes.fr/hal-02296513. It contains a notebook with short usage demonstration 'atol-demo.ipynb'. Install with:

	$ git clone https://github.com/martinroyer/atol
	$ cd atol
	$ (sudo) pip install .


# Minimal working example

```
import numpy as np
a, b, c = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]]), np.array([[1.1, 2.2, 3.3], [4.4, 5.5, 6.6], [7.7, 8.8, 9.9]]), np.array([[-4, -5, -6], [-7, -8, -9]])

from sklearn.cluster import KMeans
from atol import Atol
atol_vectoriser = Atol(quantiser=KMeans(n_clusters=10))

atol_vectoriser.fit(X=[a, b, c])

atol_vectoriser.transform(X=[a, b, c])
atol_vectoriser(a)
atol_vectoriser(c)
```

# Graph dependencies

For experiments on graphs we use the data folder and functions from the 'perslay' repository: https://github.com/MathieuCarriere/perslay, so this package should be installed at the same level as atol for instance.
