import numpy as np
import pandas as pd

# Calculate Atmira-stock-prediction metric:
def atmira_metric(y_test, y_pred):
    # Compute RRMSE
    RRMSE2 = 0.0
    for i in range(len(y_test)):
        RRMSE2 += ((y_test[i] - y_pred[i])**2)/float(y_test[i])
    RRMSE = np.sqrt(RRMSE2/len(y_test))
    # Compute number of successful cases: n_CF
    n_CF = 0
    for i in range(len(y_test)):
        if y_pred[i] >= y_test[i]:
            n_CF += 1
    CF = n_CF/len(y_test)
    asp_metric = (0.7*RRMSE) + (0.3*(1-CF))
    # Return metric
    return asp_metric

# Make scorer and define that lower scores are better
# score = make_scorer(custom_metric, greater_is_better=False)

# Apply custom scorer to ridge regression
# score(model, X_test, y_test)
