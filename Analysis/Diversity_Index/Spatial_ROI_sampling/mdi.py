#!/usr/bin/env python3
import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
import os

def compute_mdi(csv_path, step_size, output_dir="mdi_results"):
    os.makedirs(output_dir, exist_ok=True)

    # Load Shannon results
    df = pd.read_csv(csv_path)

    # Identify Shannon columns (start with "step_")
    shannon_cols = [col for col in df.columns if col.startswith("step_")]
    num_steps = len(shannon_cols)
    radii = np.arange(1, num_steps + 1) * step_size
    log_r = np.log(radii).reshape(-1, 1)

    # =============================
    # 1. GLOBAL MDI (average Shannon)
    # =============================
    mean_shannon = df[shannon_cols].mean(axis=0).values
    model_global = LinearRegression().fit(log_r, mean_shannon)
    mdi_global = model_global.coef_[0]

    print(f"Global MDI: {mdi_global:.4f}")

    # Save global summary CSV
    pd.DataFrame({
        "radius": radii,
        "log_radius": log_r.flatten(),
        "mean_shannon": mean_shannon,
        "fitted": model_global.predict(log_r)
    }).to_csv(os.path.join(output_dir, "mdi_summary_global.csv"), index=False)

    # Global plot
    plt.figure(figsize=(6, 4))
    plt.scatter(log_r, mean_shannon, label="Average Shannon")
    plt.plot(log_r, model_global.predict(log_r), color="red",
             label=f"Linear fit (MDI={mdi_global:.3f})")
    plt.xlabel("log(radius)")
    plt.ylabel("Average Shannon diversity")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, "mdi_plot_global.png"), dpi=300)
    plt.close()

    # =============================
    # 2. MDI PER CENTROID
    # =============================
    mdi_results = []
    for idx, row in df.iterrows():
        centroid = row["centroid"]
        shannon_values = row[shannon_cols].values.astype(float)

        model = LinearRegression().fit(log_r, shannon_values)
        mdi = model.coef_[0]

        mdi_results.append({"centroid": centroid, "MDI": mdi})

        # Individual plot
        plt.figure(figsize=(6, 4))
        plt.scatter(log_r, shannon_values, label="Shannon")
        plt.plot(log_r, model.predict(log_r), color="red", label=f"MDI={mdi:.3f}")
        plt.xlabel("log(radius)")
        plt.ylabel("Shannon diversity")
        plt.title(f"Centroid {centroid}")
        plt.legend()
        plt.tight_layout()
        plt.savefig(os.path.join(output_dir, f"mdi_plot_centroid_{centroid}.png"), dpi=300)
        plt.close()

    # =============================
    # 3. Save global + centroid MDI results
    # =============================
    df_mdi = pd.DataFrame(mdi_results)
    df_mdi.loc[len(df_mdi.index)] = ["GLOBAL", mdi_global]
    out_csv = os.path.join(output_dir, "mdi_all.csv")
    df_mdi.to_csv(out_csv, index=False)

    print(f"Results saved in {output_dir}")
    print(f"   - Global MDI: {mdi_global:.4f}")
    print(f"   - Per-centroid MDI values stored in mdi_all.csv")


if __name__ == "__main__":
    # === Configure here ===
    INPUT_CSV = "results_shannon.csv"  # your input CSV with Shannon indices
    STEP_SIZE = 10                     # the same step size used in your analysis
    OUTPUT_DIR = "mdi_results"

    compute_mdi(INPUT_CSV, STEP_SIZE, OUTPUT_DIR)
