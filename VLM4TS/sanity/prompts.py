SANITY1_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis.

Task: Determine whether the relationship between Channel A and Channel B is MAINTAINED throughout the entire plotted interval, or whether it BREAKS DOWN at some point and does not (or does not fully) recover.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Answer with exactly one of: "maintained" or "broken".

Respond with only the following JSON object, no other text:
{"answer": "maintained" | "broken", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


SANITY7_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis.

Task: Determine whether there is an anomalous interval in this plot, or whether both channels behave normally throughout the entire plotted interval.

Answer with exactly one of: "maintained" or "broken".

Respond with only the following JSON object, no other text:
{"answer": "maintained" | "broken", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


SANITY3_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis from time step 0 to 299.

Task: Determine whether the relationship between Channel A and Channel B is MAINTAINED throughout the entire plotted interval, or whether it BREAKS DOWN. If it breaks down, estimate the time-step interval where the relationship is broken.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Use the x-axis time steps. If the relationship is maintained, set break_start and break_end to null. If the relationship is broken, set break_start and break_end to integer time steps with 0 <= break_start < break_end <= 300. Treat break_end as an exclusive endpoint.

Respond with only the following JSON object, no markdown and no extra text:
{"answer": "maintained" | "broken", "break_start": <integer or null>, "break_end": <integer or null>, "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


SANITY4_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis from time step 0 to 299. A red shaded region marks a candidate interval proposed by an anomaly detector.

Task: Decide whether the red candidate interval is a VALID relationship-break candidate.

Use "valid" if the relationship between Channel A and Channel B is broken inside the red shaded candidate interval. Use "invalid" if the relationship is maintained inside the red interval, even if a relationship break appears elsewhere outside the red interval.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Respond with only the following JSON object, no markdown and no extra text:
{"verdict": "valid" | "invalid", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


SANITY5_PROMPT = """You are shown two stacked panels sharing the same time axis (0 to 299).

Top panel: two time-series channels, "Channel A" and "Channel B", where their relationship breaks down somewhere in the plotted interval and does not (or does not fully) recover. Vertical dashed lines mark candidate boundary time steps:
- Purple lines labeled L0, L1, L2, L3 are candidates for where the relationship break STARTS.
- Green lines labeled R0, R1, R2, R3 are candidates for where the relationship break ENDS (recovers), with R candidates always occurring after L candidates.

Bottom panel: a rolling correlation curve between Channel A and Channel B over the same time axis, with a dotted horizontal reference line, to help you see where synchronization drops and recovers.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Task: Pick exactly one candidate from {"L0", "L1", "L2", "L3"} for where the break most likely starts, and exactly one candidate from {"R0", "R1", "R2", "R3"} for where it most likely ends.

Respond with only the following JSON object, no markdown and no extra text:
{"left_option": "L0" | "L1" | "L2" | "L3", "right_option": "R0" | "R1" | "R2" | "R3", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


SANITY5B_PROMPT = """You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis from time step 0 to 299, where their relationship breaks down somewhere in the plotted interval and does not (or does not fully) recover. Vertical dashed lines mark candidate boundary time steps:
- Purple lines labeled L0, L1, L2, L3 are candidates for where the relationship break STARTS.
- Green lines labeled R0, R1, R2, R3 are candidates for where the relationship break ENDS (recovers), with R candidates always occurring after L candidates.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Task: Pick exactly one candidate from {"L0", "L1", "L2", "L3"} for where the break most likely starts, and exactly one candidate from {"R0", "R1", "R2", "R3"} for where it most likely ends.

Respond with only the following JSON object, no markdown and no extra text:
{"left_option": "L0" | "L1" | "L2" | "L3", "right_option": "R0" | "R1" | "R2" | "R3", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}"""


def build_sanity5c_prompt(left_candidates: dict, right_candidates: dict) -> str:
    left_desc = ", ".join(f"{name}={pos}" for name, pos in left_candidates.items())
    right_desc = ", ".join(f"{name}={pos}" for name, pos in right_candidates.items())
    return f"""You are shown a plot with two time-series channels, "Channel A" and "Channel B", plotted over the same time axis from time step 0 to 299, where their relationship breaks down somewhere in the plotted interval and does not (or does not fully) recover. The plot has NO markings on it -- you must judge where the break starts and ends purely by visually inspecting the two channel shapes.

Candidate time steps for where the break STARTS: {left_desc}.
Candidate time steps for where the break ENDS (recovers): {right_desc}.

"Relationship" refers to how the two channels move together (in sync, in a stable proportional or phase relationship) versus independently or in conflict, not to the absolute value of either signal.

Task: Based on visually inspecting the plot, pick exactly one label from {{"L0", "L1", "L2", "L3"}} whose time step is closest to where the break most likely starts, and exactly one label from {{"R0", "R1", "R2", "R3"}} whose time step is closest to where it most likely ends.

Respond with only the following JSON object, no markdown and no extra text:
{{"left_option": "L0" | "L1" | "L2" | "L3", "right_option": "R0" | "R1" | "R2" | "R3", "reason": "<1-3 sentences>", "confidence": <0.00-1.00>}}"""


def build_sanity6_prompt(pair_names: list[str]) -> str:
    n_channels = len(pair_names) * 2
    pair_desc = ", ".join(f"({name}: {name}a & {name}b)" for name in pair_names)
    pair_schema = ", ".join(f'"{name}": "maintained" | "broken"' for name in pair_names)
    return f"""You are shown a plot with {n_channels} time-series channels plotted over the same time axis, each with a distinct color shown in the legend.

The channels are grouped into {len(pair_names)} synchronized pairs: {pair_desc}. Each pair's two channels are supposed to move together (in sync, in a stable proportional or phase relationship).

Task: For EACH pair, determine whether the relationship between its two channels is MAINTAINED throughout the entire plotted interval, or whether it BREAKS DOWN at some point and does not (or does not fully) recover. Judge each pair independently using only that pair's two channels; a break in one pair does not imply anything about other pairs.

"Relationship" refers to how the two channels in a pair move together, not to the absolute value of either signal, and not to how one pair's channels compare to another pair's channels.

Respond with only the following JSON object, no markdown and no extra text:
{{"pairs": {{{pair_schema}}}, "reason": "<1-3 sentences covering all pairs>", "confidence": <0.00-1.00>}}"""


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
