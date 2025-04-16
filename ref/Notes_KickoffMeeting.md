## THESIS OBJECTIVES
### Dataset
 - ambiguous queries (academic writing)
 - multiple interpretations per query and their corresponding papers
 - include cases where other systems (Google Scholar, OpenScholar) fail

### Disambiguation System
 - sample interpretations per query
 - generate clarifying questions (which are optimal according to BED)
 - the answers to the clarifying questions are used to narrow down the search space

## PAPERS
### Active Task Disambiguation with LLMs
 - addresses ambiguous task specification (where information is missing and/or the intended behaviour is not clear)
 - iterative task elicitation: (iterative) interactive dialogue between user and LLM with the goal of extracting additional information from the user during each round.
 - BED: experiment optimal <=> experiment maximizes information gain about an unknown quantity of interest
 - Problem Solving Agent (LLM)
     - given a problem statement, samples candidate solutions
     - generates candidate questions that differentiate candidate solutions
     - presents the candidate question with the highest information gain to the oracle
     - updates the problem statement based on the oracle answer
 - Oracle (User)
     - provides the problem statement
     - produces oracle answer when presented with a candidate question
 - LLM agent: GPT-4o-mini, GPT-3.5-turbo, Llama-3-8B, Llama-3-70B
 - Oracle: another LLM, not specified

 ### ScholarCopilot


 ### Reading List
 - Rainforth et al. 2023 Modern Bayesian Experimental Design