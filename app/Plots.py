# =========================
# ALL-IN-ONE FASTAPI + PLOTS
# =========================

from dataclasses import dataclass
import os
import matplotlib.pyplot as plt
from fastapi import FastAPI
from fastapi.staticfiles import StaticFiles
import pandas as pd
import uuid
from datetime import datetime


@dataclass
class PlotResult:
    def generate_plot_name(self, name):
        """Generate a unique plot filename."""
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        uuid_id = str(uuid.uuid4())[:8]
        return f"{name}_{timestamp}_{uuid_id}.png"


    def plot_distribution(self, df, col, title, xlabel, ylabel, save_path):
        counts = (df[col].astype("category").value_counts(sort=False))

        labels = [str(cls).replace("_", " ").title() for cls in counts.index]
        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(counts))]

        plt.figure()
        bars = plt.bar(labels, counts.values, color=colors)

        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width() / 2, height, f"{int(height)}", ha="center", va="bottom")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()

    def plot_confidence(self, df, col, proba_col, title, xlabel, ylabel, save_path):
        mean_conf = df.groupby(col)[proba_col].mean() * 100
        labels = [cls.replace("_", " ").title() for cls in mean_conf.index]

        cmap = plt.get_cmap("tab10")
        colors = [cmap(i) for i in range(len(mean_conf))]

        plt.figure()
        plt.barh(labels, mean_conf.values, color=colors)

        for i, v in enumerate(mean_conf.values):
            plt.text(v + 1, i, f"{v:.2f}%", va="center")

        plt.xlabel(xlabel)
        plt.ylabel(ylabel)
        plt.title(title)
        plt.xlim(0, 100)

        os.makedirs(os.path.dirname(save_path), exist_ok=True)
        plt.savefig(save_path, bbox_inches="tight")
        plt.close()