from tueplots.constants.color import rgb, palettes
import matplotlib.pyplot as plt


def load_results(path):
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


def get_next_tue_plot_color(idx, mod=1.0):
    """continuous tue_plot color selector"""
    try:
        return palettes.tue_plot[idx]*mod
    except IndexError:
        return get_next_tue_plot_color(idx-len(palettes.tue_plot), mod*1.2)


if __name__ == "__main__":
    SAVE_DIR = "../latex/msc_thesis/images/plot_"

    # retrieval
    # k = 10, 5, 1
    sc_retr_eval_withoutAbs = {
        "intro+relwork": [0.126, 0.114, 0.069],
        "methods": [0.185, 0.16, 0.14],
        "experiments": [0.33, 0.291, 0.243],
        "conclusion": [0.147, 0.121, 0.072]
    }
    pr_retr_eval_withoutAbs = {
        "intro+relwork": [0.092, 0.083, 0.062],
        "methods": [0.149, 0.132, 0.115],
        "experiments": [0.256, 0.218, 0.203],
        "conclusion": [0.117, 0.083, 0.066]
    }
    sc_retr_eval_abs = {
        "intro+relwork": [0.121, 0.121, 0.067],
        "methods": [0.178, 0.154, 0.119],
        "experiments": [0.315, 0.275, 0.245],
        "conclusion": [0.147, 0.099, 0.086]
    }
    pr_retr_eval_abs = {
        "intro+relwork": [0.104, 0.099, 0.06],
        "methods": [0.148, 0.121, 0.106],
        "experiments": [0.242, 0.225, 0.211],
        "conclusion": [0.124, 0.079, 0.079]
    }
    # intro+relwork, methods, experiments, conclusion
    sc_reasonir_withoutAbs = [0.139, 0.158, 0.316, 0.133]
    pr_reasonir_withoutAbs = [0.075, 0.093, 0.195, 0.08]

    plot_data_container = {
        "masked": [
            sc_retr_eval_withoutAbs,
            pr_retr_eval_withoutAbs
        ],
        "with abstracts": [
            sc_retr_eval_abs,
            pr_retr_eval_abs
        ]
    }

    # retrival plots
    FONTSIZE = 10
    MS = 2
    LW = 1.3
    GRID_LW = 0.5
    AXES_ASPECT = 10

    recall_labels = [f"Recall@{k}" for k in [10, 5, 1]]
    for replacement_strategy in plot_data_container.keys():
        fig, ax = plt.subplots(1, 2, sharey=True)
        for i, key in enumerate(["intro+relwork", "methods", "experiments", "conclusion"]):
            ax[0].plot(
                recall_labels,
                plot_data_container[replacement_strategy][0][key],
                "-",
                ms=MS,
                lw=LW,
                color=get_next_tue_plot_color(i),
                label=key
            )
            ax[0].set_title(f"SC {replacement_strategy}")
            ax[1].plot(
                recall_labels,
                plot_data_container[replacement_strategy][1][key],
                "-",
                ms=MS,
                lw=LW,
                color=get_next_tue_plot_color(i),
                label=key
            )
            ax[1].set_title(f"SC+PR {replacement_strategy}")

            ax[0].set_ylabel("Recall %", fontsize=FONTSIZE)
            ax[1].legend(bbox_to_anchor=(1.01, 1)).get_frame().set_edgecolor(color=rgb.tue_gray)
            for j in [0, 1]:
                ax[j].set_aspect(AXES_ASPECT)
                ax[j].grid(axis="both", color=rgb.tue_gray, linewidth=GRID_LW)

        fig.tight_layout()

        fig.savefig(SAVE_DIR+f"retrieval_{replacement_strategy.replace(" ", "_")}.pdf")
    

    # generation
    