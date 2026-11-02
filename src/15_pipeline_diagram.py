"""Step 15 - Pipeline diagram for the final report (Figure 1 of the Methodology): Data -> Preprocessing -> EDA ->
Feature engineering -> Model -> Deployment, with the script that implements each stage. Writes outputs/report/fig_pipeline.png."""
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Patch
import plot_style as ps
from config import OUTPUTS

TOOL = {"Python": ps.NAVY, "SQLite": "#2E8B57", "SPSS": ps.ORANGE, "Streamlit": "#7B4EA3"}
STAGES = [
    ("1. Data", "SQLite", ["NSS 80th round (2025)", "NSS 75th round (2017-18)", "Cancer episodes (code 13)", "N = 3,283 and 1,606", "01, 10"]),
    ("2. Preprocessing", "Python", ["Merge household, person,", "hospital files", "Missing values, zero costs,", "outliers, scaling, encoding", "01, 11"]),
    ("3. EDA", "Python", ["Weighted descriptives", "Distributions, group profiles", "Correlations, VIF", "Figures 2-7", "06"]),
    ("4. Feature engineering", "Python", ["14 features, 2 targets", "Within-survey quintiles", "ln(OOPE + 1), CHE flag", "No cost leakage", "01, 11"]),
    ("5. Model", "Python", ["Linear, RF, LightGBM,", "XGBoost, neural net", "80/20 split by FSU, tuning", "Logistic (RQ2), SPSS (RQ2-4)", "03, 11, 13"]),
    ("6. Deployment", "Streamlit", ["Cost range + CHE flag", "Top drivers + aid checklist", "Saved models, retrain", "when a new NSS round", "14, app/"]),
]


def main():
    fig, ax = plt.subplots(figsize=(13.4, 3.9)); ax.set_xlim(0, 13.4); ax.set_ylim(0, 3.9); ax.axis("off")
    w, h, gap, y0 = 1.95, 2.55, 0.26, 0.75
    for i, (title, tool, lines) in enumerate(STAGES):
        x = 0.12 + i * (w + gap)
        ax.add_patch(FancyBboxPatch((x, y0), w, h, boxstyle="round,pad=0.02,rounding_size=0.08", fc="white", ec=TOOL[tool], lw=2))
        ax.add_patch(FancyBboxPatch((x, y0 + h - 0.5), w, 0.5, boxstyle="round,pad=0.02,rounding_size=0.08", fc=TOOL[tool], ec=TOOL[tool]))
        ax.text(x + w / 2, y0 + h - 0.25, title, ha="center", va="center", color="white", fontsize=8.8, fontweight="bold")
        for j, t in enumerate(lines[:-1]):
            ax.text(x + w / 2, y0 + h - 0.85 - j * 0.34, t, ha="center", va="center", fontsize=7.4)
        ax.text(x + w / 2, y0 + 0.17, f"scripts: {lines[-1]}", ha="center", va="center", fontsize=7.2, style="italic", color="#555555")
        if i < len(STAGES) - 1:
            ax.add_patch(FancyArrowPatch((x + w + 0.02, y0 + h / 2), (x + w + gap - 0.02, y0 + h / 2), arrowstyle="-|>", mutation_scale=14, color="#444444", lw=1.5))
    # feedback arrow: retraining
    ax.add_patch(FancyArrowPatch((0.12 + 5 * (w + gap) + w / 2, y0 - 0.05), (0.12 + w / 2, y0 - 0.05), connectionstyle="arc3,rad=-0.12",
                                 arrowstyle="-|>", mutation_scale=14, color="#888888", lw=1.3, ls="--"))
    ax.text(6.7, 0.38, "Retrain when a new NSS round is released", ha="center", fontsize=8, color="#666666")
    ax.legend(handles=[Patch(fc=c, ec=c, label=k) for k, c in TOOL.items()], loc="upper center", ncol=4, frameon=False, fontsize=8.5,
              title="Main tool in each stage", title_fontsize=8.5, bbox_to_anchor=(0.5, 1.12))
    ps.save(fig, OUTPUTS / "report" / "fig_pipeline.png")
    print("wrote outputs/report/fig_pipeline.png")


if __name__ == "__main__":
    main()
