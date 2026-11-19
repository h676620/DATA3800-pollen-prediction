import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

SERVICE = "http://127.0.0.1:8000"
MODEL = "gemini-3.5-flash-lite"


# Every field may come back empty, and there is one extra field for the
# model to ask a question when it does not know something yet.
class Car(BaseModel):
    year: int | None = None
    present_price: float | None = None
    kms_driven: int | None = None
    owner: int | None = None
    fuel_type: str | None = None
    seller_type: str | None = None
    transmission: str | None = None
    brand: str | None = None
    question: str | None = None


# Google answers 503 when a model is busy. The library does not retry
# unless you ask it to, so ask it to.
client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=5)
    )
)

config = types.GenerateContentConfig(
    system_instruction=(
        "You price second hand cars. Fill in what the person has told "
        "you and leave everything else null. Never guess a value. "
        "Categories are capitalised as Petrol, Diesel, CNG, Dealer, "
        "Individual, Manual, Automatic. The brand is the model name in "
        "lower case, so city for a Honda City. present_price is what "
        "the car cost when it was new, not what it is worth now or "
        "what it is being sold for. Assume 0 previous owners unless "
        "told otherwise. If any field is still null, put one short "
        "question in 'question' asking for the single most important "
        "thing you still need, and ask for nothing else. If you "
        "already asked for that same thing and did not get it, ask "
        "again in different words and say why you need it. "
        "When every field is filled, leave 'question' null. On a "
        "follow up, keep everything you already know and change only "
        "what the person just said."
    ),
    response_mime_type="application/json",
    response_schema=Car,
)

history = []


def price(question_text):
    history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=question_text)],
        )
    )

    reply = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=config,
    )
    history.append(reply.candidates[0].content)

    car = Car.model_validate_json(reply.text)
    fields = car.model_dump(exclude={"question"})

    if car.question or any(value is None for value in fields.values()):
        return car, None  # Not enough to price yet.

    answer = requests.post(
        f"{SERVICE}/predict",
        json=fields,
        timeout=10,
    )
    answer.raise_for_status()
    return car, answer.json()


if __name__ == "__main__":
    print("Tell me about the car. Empty line to stop.\n")

    while True:
        said = input("you: ")

        if not said.strip():
            break

        car, answer = price(said)
        shown = car.model_dump(exclude={"question"})
        known = {key: value for key, value in shown.items() if value is not None}
        missing = [key for key, value in shown.items() if value is None]

        print(" so far: ", known)

        if missing:
            print(" missing:", ", ".join(missing))

        if answer is None:
            print("bot: ", car.question or "I need a little more.")
            print()
        else:
            print(
                "bot: that comes out at",
                answer["selling_price_lakhs"],
                "lakhs",
            )
            print()