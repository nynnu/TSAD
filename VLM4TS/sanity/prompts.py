SANITY1_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis.

Task: Determine whether the relationship between Channel A and Channel B is MAINTAINED throughout the entire plotted interval, or whether it BREAKS DOWN at some point and does not (or does not fully) recover.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Answer with exactly one of: "maintained" or "broken".

Respond with only the following JSON object, no other text:
{"answer": "maintained" | "broken", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


def build_sanity2_judge_prompt(model_answer: str, model_reason: str) -> str:
    return f"""You are evaluating another vision-language model's explanation for a two-channel time-series synchronization task.

You are shown the same plot that the model saw. The original task was to decide whether the relationship between Channel A and Channel B was "maintained" or "broken".

The model answered:
answer: {model_answer}
reason: {model_reason}

Classify the reason into exactly one category:

1. "relational": The reason explicitly compares Channel A and Channel B, or describes their synchronization, phase relationship, proportional relationship, divergence, alignment, or joint movement.
2. "single_channel": The reason mainly describes an abnormality in one channel alone without using the relation between the two channels as evidence.
3. "vague": The reason is too generic or underspecified to tell what visual evidence was used.
4. "hallucinated": The reason cites visual evidence that is not present in the plot.

Use the image to check whether the reason is visually supported. Do not judge whether the original answer is correct; only classify the reasoning style.

Respond with only this JSON object, no markdown and no extra text:
{{"reason_type": "relational" | "single_channel" | "vague" | "hallucinated", "rationale": "<one short sentence>"}}"""
