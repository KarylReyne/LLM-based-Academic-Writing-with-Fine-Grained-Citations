

def load_results(path):
    results = {}
    with open(path, 'r', encoding='utf-8') as f:
        results = json.load(f)
    return results


if __name__ == "__main__":
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
        "experiments": [0.242, 0.225, 0.0.211],
        "conclusion": [0.124, 0.079, 0.079]
    }
    # intro+relwork, methods, experiments, conclusion
    sc_reasonir_withoutAbs = [0.139, 0.158, 0.316, 0.133]
    pr_reasonir_withoutAbs = [0.075, 0.093, 0.195, 0.08]

    # generation
    