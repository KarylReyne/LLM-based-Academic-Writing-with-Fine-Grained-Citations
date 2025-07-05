### Required
#### pre-thesis
 - passage retrieval evaluation on fulltext dataset (recall@k of gold citation given written paper context)
 - whole model generation quality evaluation, better or worse than standard SC? (use eval prompt in SC paper, maybe use SC corpus by expanding it with fulltexts?)
 - passage retrieval finetuning on fulltext dataset (contrastive learning on written papers, negative via in-batch sampling? -> see SC)
 - query reranker for reasoning why the passages are ranked this way in relation to each other (during ranking LLM call or as separat LLM call?)
#### post-thesis
 - generation context window
 - hyperparameter tuning (e.g. batched self-consistency yes/no)

### tmux
tmux new -s SESSION_NAME
CTRL+B+D (window -> bg process)
tmux attach -t SESSION_NAME
