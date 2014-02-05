import numpy as np
from statsmodels.distributions.empirical_distribution import ECDF

ecdf = ECDF([3, 3, 1, 4])
ecdf([3, 55, 0.5, 1.5])