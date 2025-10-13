

# retriever instructions (based on ReasonIR / BRIGHT)
retrieval_instruction_query = "<|user|>\nGiven a query with a citation marked by <|cite_start|>, retrieve relevant passages that describe the cited topic\n<|embed|>\n"
retrieval_instruction_document = f"<|embed|>\n"

scores = "score" if len(d)<2 else "scores"
paragraphs = "".join([f"<start_paragraph-{i+1}>{p}<end_paragraph-{i+1}>" for i, p in enumerate(d)])
reranking_instruction = lambda q, d: f"You are given a query and {len(d)} paragraphs. Assign each paragraph a score based how likely it is for the given paragraph to be cited at the end of the query. A paragraph should receive a high score if it explains the same or a topic similar to what is discussed towards the end of the query. Following the order of the passages below, your answer should be '[..., xi, ...]' where each xi is a number from 0-10 and the score of the corresponding paragraph-i. 0 means completely irrelevant, 10 means highly relevant and very likely to be cited at the end of the query. Don't output anything else. Output exactly {len(d)} {scores}. Here is the query: <start_query>{q}<end_query>Here are the paragraphs: {paragraphs}"