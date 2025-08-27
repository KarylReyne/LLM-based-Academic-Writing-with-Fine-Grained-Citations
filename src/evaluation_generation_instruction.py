import random
import numpy as np

judge_instruction = lambda t, a, gold, gen: f"""
    You are a senior computer science scholar. Please evaluate the AI-generated content
    using the ground truth as reference.
    Evaluate the following five dimensions by comparing the AI-generated content with the
    ground truth:
    [Detailed Evaluation]
    1. Content Relevance:
    - Key strengths:
    - Main gaps:
    - Comparison with ground truth:
    2. Logical Coherence:
    - Key strengths:
    - Main gaps:
    - Comparison with ground truth:
    3. Academic Standards:
    - Key strengths:
    - Main gaps:
    - Comparison with ground truth:
    4. Background Completeness:
    - Key strengths:
    - Main gaps:
    - Comparison with ground truth:
    5. Innovation Statement:
    - Key strengths:
    - Main gaps:
    - Comparison with ground truth:
    [End Evaluation]
    [Improvement Suggestions]
    1.
    2.
    3.
    [End Suggestions]
    Based on your above analysis, provide numerical scores in the following format:
    [Scores]
    Relevance: <score>/5
    Coherence: <score>/5
    Academic: <score>/5
    Completeness: <score>/5
    Innovation: <score>/5
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
    Remember to first provide detailed evaluation, then improvement suggestions, and
    finally the numerical scores in the exact format specified above.
"""

def judge_instruction2(t, a, gold, gen, shuffle=True):
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
    
    instruction_begin = """
        You are a senior computer science scholar. Please evaluate the AI-generated content
        using the ground truth as reference.
        Evaluate the following five dimensions by comparing the AI-generated content with the
        ground truth:
        [Detailed Evaluation]"""
    
    instruction_interm = """
        [End Evaluation]
        [Improvement Suggestions]
        1.
        2.
        3.
        [End Suggestions]
        Based on your above analysis, provide numerical scores in the following format:
        [Scores]"""
    
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
        Remember to first provide detailed evaluation, then improvement suggestions, and
        finally the numerical scores in the exact format specified above.
        Make sure that you output the numerical scores last."""

    instruction = instruction_begin
    j = 1
    for i in indices:
        instruction += eval_tasks(j, eval_categories[i])
        j += 1
    instruction += instruction_interm
    for i in indices:
        instruction += score_format(eval_categories[i])
    instruction += instruction_end
    return instruction