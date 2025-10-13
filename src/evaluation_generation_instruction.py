import random
import numpy as np

def judge_instruction2(t, a, gold, gen, shuffle=True, only_scores=False):
    indices = np.arange(5)
    if shuffle:
        random.shuffle(indices)

    eval_categories = [
        "Content Relevance",
        "Logical Coherence",
        "Academic Rigor",
        "Background Completeness",
        "Innovation Statement"
    ]

    eval_tasks = lambda i, l: f"""
        {i}. {l}:
        - Key strengths:
        - Main gaps:
        - Comparison with ground truth:"""

    score_format = lambda l: f"""
        {l}: <score>/5"""
    
    insert = "Evaluate the following five dimensions by comparing the AI-generated content with the ground truth." if only_scores else "Evaluate the following five dimensions by comparing the AI-generated content with the ground truth:\n[Detailed Evaluation]"
    instruction_begin = f"""
        You are a senior computer science scholar. Please evaluate the AI-generated content
        using the ground truth as reference.
        {insert}"""
    
    insert = "Based on your analysis, provide numerical scores in the following format:" if only_scores else """
        [End Evaluation]
        [Improvement Suggestions]
        1.
        2.
        3.
        [End Suggestions]
        Based on your above analysis, provide numerical scores in the following format:"""
    instruction_interm = f"""
        {insert}
        [Scores]"""
    
    insert = "Remember to provide the numerical scores in the exact format specified above.\nMake sure that you output the numerical scores last." if only_scores else "Remember to first provide detailed evaluation, then improvement suggestions, and finally the numerical scores in the exact format specified above.\nMake sure that you output the numerical scores last."
    instruction_end = f"""
        Total: <sum>/25
        [End Scores]
        Below are the materials for evaluation:
        Paper Title:
        {t}
        Abstract:
        {a}
        Ground Truth Content:
        {gold}
        AI Generated Content:
        {gen}
        {insert}"""

    instruction = instruction_begin
    j = 1
    if not only_scores:
        for i in indices:
            instruction += eval_tasks(j, eval_categories[i])
            j += 1
    instruction += instruction_interm
    for i in indices:
        instruction += score_format(eval_categories[i])
    instruction += instruction_end
    return instruction