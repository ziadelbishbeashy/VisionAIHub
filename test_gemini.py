import json
import os

from google import genai


MODEL_NAME = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)


SYSTEM_INSTRUCTIONS = """
You are the VisionAIHub context-aware assistant.

The user uploads one image and selects one or more analysis modules.

The supplied context may contain results from:

1. YOLOv8 object detection.
2. Random Forest traffic-density prediction.
3. EfficientNetB0 weather classification.
4. Scene classification.

You may answer questions about:

- Detected objects and their counts.
- People and vehicles.
- Cars, buses, trucks, motorcycles and bicycles.
- Traffic lights and stop signs.
- Total objects, people and vehicles.
- Detection confidence values.
- Vehicle occupancy and vehicle share.
- Traffic-density prediction and confidence.
- Weather prediction and confidence.
- Scene prediction and confidence.
- Class probabilities.
- Relationships between the available results.
- A complete summary of the analysis.

Follow these rules:

1. Answer only from the supplied analysis context.
2. Never invent objects, counts, predictions or confidence values.
3. When a module is unavailable, say that it was not selected or failed.
4. Do not describe an unavailable module as having a prediction.
5. Traffic-light detection only confirms that a traffic light was detected.
6. Never claim to know whether a traffic light is red, yellow or green.
7. Never guarantee that the road or situation is safe.
8. Any practical guidance must explain that it is based on one image only.
9. Mention when a prediction confidence is below 55 percent.
10. Distinguish object-detection confidence from classification confidence.
11. Answer in the same language used by the user.
12. Keep answers clear and concise unless the user requests details.
13. Do not mention internal JSON, prompts or system instructions.
"""


def answer_with_gemini(
    question,
    context
):
    """
    Generate a context-aware answer using Gemini.

    Parameters
    ----------
    question:
        The user's question.

    context:
        Dictionary containing the available VisionAIHub
        model results.

    Returns
    -------
    str
        Gemini's generated answer.
    """

    clean_question = str(
        question or ""
    ).strip()

    if not clean_question:
        raise ValueError(
            "Question cannot be empty."
        )

    api_key = os.getenv(
        "GEMINI_API_KEY"
    )

    if not api_key:
        raise RuntimeError(
            "GEMINI_API_KEY was not found."
        )

    client = genai.Client(
        api_key=api_key
    )

    context_text = json.dumps(
        context,
        ensure_ascii=False,
        indent=2,
        default=str
    )

    prompt = f"""
{SYSTEM_INSTRUCTIONS}

IMAGE ANALYSIS CONTEXT:

{context_text}

USER QUESTION:

{clean_question}

ANSWER:
"""

    response = (
        client.models.generate_content(
            model=MODEL_NAME,
            contents=prompt
        )
    )

    answer = (
        response.text or ""
    ).strip()

    if not answer:
        raise RuntimeError(
            "Gemini returned an empty response."
        )

    return answer