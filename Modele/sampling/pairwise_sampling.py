def pairwise_sampling(df, target_column, config):
    import pandas as pd
    import numpy as np
    from allpairspy import AllPairs

    # Définir le nombre de bins par feature
    num_bins = config["sampling"]["pairwise"].get("num_bins", 5)
    
    # Séparer X et y
    X = df.drop(columns=[target_column])
    y = df[target_column]

    # Binnage automatique
    bins_dict = {}
    for feature in X.columns:
        min_val = X[feature].min()
        max_val = X[feature].max()
        bins = np.linspace(min_val, max_val, num_bins + 1)
        bins_dict[feature] = bins.tolist()

    all_samples = []

    for class_label in y.unique():
        class_df = df[df[target_column] == class_label].copy()
        for feature, bins in bins_dict.items():
            labels = list(range(1, len(bins)))
            class_df[feature] = pd.cut(class_df[feature], bins=bins, labels=labels, include_lowest=True)
        
        param_values = {
            feature: class_df[feature].dropna().unique().tolist()
            for feature in X.columns
        }

        pairs = list(AllPairs(param_values.values()))
        df_pairwise = pd.DataFrame(pairs, columns=param_values.keys())
        df_pairwise[target_column] = class_label
        all_samples.append(df_pairwise)

    df_sampled = pd.concat(all_samples, ignore_index=True)
    return df_sampled
