import numpy as np
from sklearn.cross_validation import cross_val_score

def unbalanced_cross_val(model, x, y, scoring='f1', folds=10):
    if y.mean() > 0.5:
        unbal_class = 1
        bal_class = 0
    else:
        unbal_class = 0
        bal_class = 1

    n_total = len(y)
    n_maj = len(y[y == unbal_class])
    n_min = n_total - n_maj
    n_splits = n_maj / n_min

    ind_maj = np.where(y == unbal_class)[0]
    ind_min = np.where(y == bal_class)[0]

    np.random.shuffle(ind_maj)

    splits = np.array_split(ind_maj, n_splits)
    
    scores = []
    for split in splits:
        x_maj = x[split]
        x_min = x[ind_min]
        y_maj = y[split]
        y_min = y[ind_min]

        x_cv = np.append(x_maj, x_min, axis=0)
        y_cv = np.append(y_maj, y_min)
        
        score = cross_val_score(model, x_cv, y_cv, scoring=scoring, cv=folds)
        scores.append(score)

    return scores

    
