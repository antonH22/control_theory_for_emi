import sys
import os
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import numpy as np
import pandas as pd
import myplots

### Plot metrics of the LDS model across training set lengths (5.1.4, Figure 4)

def process_and_plot_all(df_path, log_metrics=None, reverse_metrics=None, ylim_dict=None):
    df = pd.read_csv(df_path)

    metrics = df["metric"].unique()
    for metric in metrics:
        sub_df = df[df["metric"] == metric]
        trainlens = sub_df["trainlen"].values
        means = sub_df["mean"].values
        ses = sub_df["se"].values
        ylabel = sub_df["ylabel"].iloc[0]

        # Optional transformations
        if reverse_metrics and metric in reverse_metrics:
            trainlens = trainlens[::-1]
            means = means[::-1]
            ses = ses[::-1]

        log_scale = log_metrics and metric in log_metrics
        ylim = ylim_dict.get(metric) if ylim_dict else None

        save_path = os.path.join(
            "results_replicated",
            f"{metric}{'_logplot' if log_scale else '_plot'}.png"
        )

        myplots.myplot_bar(
            trainlens,
            means,
            ses,
            ylim=ylim,
            log_scale=log_scale,
            xlabel='Training set length',
            ylabel=ylabel,
            title='',
            save_path=None
        )

process_and_plot_all(
    "results_replicated/results_training_set_lengths_metrics.csv",
    log_metrics=["frobenius_AC"],
    reverse_metrics=[],
    ylim_dict={"eigenvalue_A": 1.001}
)