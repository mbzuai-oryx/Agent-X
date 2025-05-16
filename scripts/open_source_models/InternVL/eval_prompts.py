








############### Grounding Score #############################

grounding_score_prompt = """You are given: \
- Ground Truth (GT) description of the image. \
- The Agent's reasoning step or tool input. \
Task: \
Check whether the agent's references to objects, regions, or attributes match what is actually present in the ground truth description. 
Score: \
- 1: All references are correctly grounded. 
- 0.5: Some references are grounded, but some are missing or wrong. \
- 0: No correct grounding; hallucinated or irrelevant elements mentioned. \
Output only the score (0, 0.5, or 1) and a one-line justification.
"""