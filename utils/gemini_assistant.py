import json
import os
import time

from google import genai


PRIMARY_MODEL = os.getenv(
    "GEMINI_MODEL",
    "gemini-2.5-flash-lite"
)

FALLBACK_MODEL = os.getenv(
    "GEMINI_FALLBACK_MODEL",
    "gemini-2.5-flash"
)

MODEL_NAMES = [
    PRIMARY_MODEL,
    FALLBACK_MODEL
]

MAX_ATTEMPTS_PER_MODEL = 2
BASE_RETRY_DELAY = 2


SYSTEM_INSTRUCTIONS = """
You are the VisionAIHub context-aware assistant.

The supplied context may contain results from:

1. YOLOv8 object detection.
2. Random Forest traffic-density prediction.
3. EfficientNetB0 weather classification.
4. Scene classification.

You may answer questions about:

- Detected objects and their counts.
- People and vehicles.
- Traffic density and confidence.
- Weather prediction and confidence.
- Scene prediction and confidence.
- Class probabilities.
- The complete image analysis.

Rules:

1. Answer only from the supplied analysis context.
2. Never invent objects, counts, predictions, or confidence values.
3. If a module is unavailable, say that it was not selected or failed.
4. Traffic-light detection only confirms its presence.
5. Never guarantee that a road or situation is safe.
6. Mention when a confidence value is below 55 percent.
7. Answer in the same language as the user.
8. Keep the answer clear and concise.
"""


def is_temporary_error(error):
    """
    Return True for temporary Gemini server errors
    that may succeed after retrying.
    """

    error_text = str(error).lower()

    temporary_markers = [
        "503",
        "504",
        "unavailable",
        "high demand",
        "overloaded",
        "temporarily",
        "deadline exceeded",
        "internal"
    ]

    return any(
        marker in error_text
        for marker in temporary_markers
    )


def answer_with_gemini(question, context):
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

    last_error = None

    for model_name in MODEL_NAMES:

        for attempt in range(
            MAX_ATTEMPTS_PER_MODEL
        ):
            try:
                print(
                    "Trying Gemini model:",
                    model_name,
                    "| Attempt:",
                    attempt + 1
                )

                response = (
                    client.models.generate_content(
                        model=model_name,
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

                print(
                    "Gemini response succeeded using:",
                    model_name
                )

                return answer

            except Exception as error:
                last_error = error

                print(
                    "Gemini request failed:",
                    model_name,
                    "|",
                    type(error).__name__,
                    "|",
                    error
                )

                if not is_temporary_error(
                    error
                ):
                    raise

                has_another_attempt = (
                    attempt
                    < MAX_ATTEMPTS_PER_MODEL - 1
                )

                if has_another_attempt:
                    delay = (
                        BASE_RETRY_DELAY
                        * (2 ** attempt)
                    )

                    print(
                        f"Retrying in {delay} seconds..."
                    )

                    time.sleep(
                        delay
                    )

    raise RuntimeError(
        "Gemini is temporarily experiencing high demand. "
        "Please wait a moment and try again."
    ) from last_error