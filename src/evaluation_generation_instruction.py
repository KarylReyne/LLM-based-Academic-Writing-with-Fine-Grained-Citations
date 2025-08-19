

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

judge_instruction2 = lambda t, a, gold, gen: f"""
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
    3. Academic Rigor:
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
    Content Relevance: <score>/5
    Logical Coherence: <score>/5
    Academic Rigor: <score>/5
    Background Completeness: <score>/5
    Innovation Statement: <score>/5
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