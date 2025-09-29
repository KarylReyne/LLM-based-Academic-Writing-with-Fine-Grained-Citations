from tueplots.constants.color import rgb, palettes
import matplotlib.pyplot as plt
import json


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
    if False: # last token in abstract
        import ijson
        from dataset_loaders import arxiv_to_corpus_id, load_retrieval_dataset, scholarcopilot_arxiv_to_corpus_id, load_sections_eval_dataset, load_scholarcopilot_metadata_corpus

        docs_id_map_path = "data/arxiv_to_corpus_id_documents_3.0.json"
        processed_corpus_path = "data/documents_3.0_processed_corpus.jsonl"
        docs_corpus_id_map = arxiv_to_corpus_id(docs_id_map_path, processed_corpus_path)

        sc_corpus_id_map_path = "data/arxiv_to_corpus_id_scholar_copilot_train_data_500k.json"
        sc_corpus_path = "scholarcopilot_data/corpus_data_arxiv_1215.jsonl"
        sc_corpus_id_map = scholarcopilot_arxiv_to_corpus_id(sc_corpus_id_map_path, sc_corpus_path)
        sc_arxiv_id_map = {v: k for k, v in sc_corpus_id_map.items()}
        sc_to_docs_corpus_id = lambda x: docs_corpus_id_map[sc_arxiv_id_map[x]]
        
        docs_retrieval_dataset_path = "data/retrieval_dataset_documents_3.0.jsonl"
        complete_dataset_path = "data/documents_3.0_with_ids.jsonl"
        docs_retrieval_dataset = load_retrieval_dataset(docs_retrieval_dataset_path, complete_dataset_path, docs_corpus_id_map)

        for path in [
            "data_thesis/eval_dataset_intro+relwork-sections_documents_3.0_for_sc_corpus.jsonl",
            "data_thesis/eval_dataset_methods-sections_documents_3.0_for_sc_corpus.jsonl",
            "data_thesis/eval_dataset_experiments-sections_documents_3.0_for_sc_corpus.jsonl",
            "data_thesis/eval_dataset_conclusion-sections_documents_3.0_for_sc_corpus.jsonl"
        ]:  
            print(path)
            print("collecting...", end="")
            count = 0
            length = 0
            with open(path, "rb") as file:
                for item in ijson.items(file, "", multiple_values=True):
                    ctx = item["context"]
                    tgt = item["target_arxiv_id"]
                    # print(f"{ctx.rstrip(" ").split(" ")[-1]} IN {docs_retrieval_dataset[docs_corpus_id_map[tgt]]["title"]}")
                    count += ctx.rstrip(" ").split(" ")[-1] in docs_retrieval_dataset[docs_corpus_id_map[tgt]]["abstract"]
                    length += 1
            print("done")
            print(f"{count/length} = {count}/{length} count/length\n")

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

    # avg recall
    avg_change_masked = 0
    avg_change_abstracts = 0
    for i in range(3):
        for key in ["intro+relwork", "methods", "experiments", "conclusion"]:
            avg_change_masked += plot_data_container["masked"][1][key][i]-plot_data_container["masked"][0][key][i]
            avg_change_abstracts += plot_data_container["with abstracts"][1][key][i]-plot_data_container["with abstracts"][0][key][i]
    avg_change_masked /= 12
    avg_change_abstracts /= 12
    print(f"avg change: {avg_change_masked}/{avg_change_abstracts} on masked/abstracts")

    # cumulative recall change
    cum_change = {}
    for replacement_strategy in plot_data_container.keys():
        d = [{"sum": 0}, {"sum": 0}]
        for j in [0, 1]: # sc, pr
            for key in ["intro+relwork", "methods", "experiments", "conclusion"]:
                d[j][key] = 0
                for i in [1, 2]:
                    change = plot_data_container[replacement_strategy][j][key][i]-plot_data_container[replacement_strategy][j][key][i-1]
                    d[j][key] += change
                    d[j]["sum"] += d[j][key]
                d[j][key] = "{0:.2f}".format(d[j][key])
            d[j]["sum"] = sum([float(d[j][key]) for key in ["intro+relwork", "methods", "experiments", "conclusion"]])
        cum_change[replacement_strategy] = d
    print(cum_change)

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
    gen_eval_reasoning_data = load_results("out_thesis/eval_generation_judge_instruction2_30000_100/2025-08-30/records_eval_generation_2025-08-30_11-24-16.json")
    gen_eval_scores_data = load_results("out_thesis/eval_generation_judge_instruction2-only-scores_30000_100/2025-08-30/records_eval_generation_2025-08-30_18-38-09.json")

    FONTSIZE = 10
    BW = 0.45
    LW = 0.2
    GRID_LW = 0.5
    AXES_ASPECT = 10

    for label, plot_data in [["with reasoning", gen_eval_reasoning_data], ["without reasoning", gen_eval_scores_data]]:
        dim_labels = list(plot_data["SC scores stats"].keys())[:-1]
        short_lbl = lambda l: "\n".join(l.split(" "))

        fig, ax = plt.subplots(figsize=(10,5))
        ax.bar(
            [short_lbl(l) for l in dim_labels],
            [plot_data["SC scores stats"][dim]["mean"] for dim in dim_labels],
            width=-BW,
            color=get_next_tue_plot_color(0),
            align="edge",
            label="SC"
        )
        ax.bar(
            [short_lbl(l) for l in dim_labels],
            [plot_data["SC PR scores stats"][dim]["mean"] for dim in dim_labels],
            width=BW,
            color=get_next_tue_plot_color(2),
            align="edge",
            label="SC+PR"
        )
        ax.set_title(f"Judge scores {label}")
        ax.set_ylabel("score (0-5)", fontsize=FONTSIZE)
        ax.set_ylim([2.5, 4])
        ax.legend(bbox_to_anchor=(0.99, 0.99)).get_frame().set_edgecolor(color=rgb.tue_gray)

        # fig.tight_layout()
        fig.savefig(SAVE_DIR+f"generation_{label.replace(" ", "_")}.pdf")


    fig, ax = plt.subplots(figsize=(10,5))
    ax.bar(
        [short_lbl(l) for l in dim_labels],
        [gen_eval_reasoning_data["SC scores stats"][dim]["std"] for dim in dim_labels],
        width=-BW,
        # linewidth=LW,
        # edgecolor="w",
        color=get_next_tue_plot_color(1),
        align="edge",
        label="with reasoning"
    )
    ax.bar(
        [short_lbl(l) for l in dim_labels],
        [gen_eval_scores_data["SC scores stats"][dim]["std"] for dim in dim_labels],
        width=BW,
        # linewidth=LW,
        # edgecolor="w",
        color=get_next_tue_plot_color(3),
        align="edge",
        label="without reasoning"
    )
    # barplots side by side with small gap/margin
    ax.set_title(f"Standard deviation of the scores for SC")
    ax.set_ylabel(r"$\delta$", fontsize=FONTSIZE)
    ax.set_ylim([0.2, 0.8])
    ax.legend(bbox_to_anchor=(0.99, 0.99)).get_frame().set_edgecolor(color=rgb.tue_gray)

    # fig.tight_layout()
    fig.savefig(SAVE_DIR+f"generation_std_{label.replace(" ", "_")}.pdf")