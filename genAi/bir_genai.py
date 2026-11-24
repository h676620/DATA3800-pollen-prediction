import requests
from google import genai
from google.genai import types
from pydantic import BaseModel
import requests
from google import genai
from google.genai import types
from pydantic import BaseModel

SERVICE = "http://127.0.0.1:8000"
MODEL = "gemini-3.5-flash-lite"


class PollenRequest(BaseModel):
    today_date: str | None = None
    pollen_type: str | None = None
    question: str | None = None


client = genai.Client(
    http_options=types.HttpOptions(
        retry_options=types.HttpRetryOptions(attempts=5)
    )
)

config = types.GenerateContentConfig(
    system_instruction=(
    "Hent ut dato og pollentype som brukeren oppgir. "
    "Datoformatet skal være DD.MM.YYYY, for eksempel 12.04.2025. "
    "Støttede pollentyper er Bjørk og Gress. "
    "Ikke finn på opplysninger. Hvis dato eller pollentype mangler, "
    "still ett kort spørsmål om én manglende opplysning. "
    "Når begge er oppgitt, skal question være null. "
    "Behold opplysninger fra tidligere meldinger i samtalen."
    ),
    response_mime_type="application/json",
    response_schema=PollenRequest,
)

history = []


def get_forecast(message: str):
    history.append(
        types.Content(
            role="user",
            parts=[types.Part.from_text(text=message)],
        )
    )

    reply = client.models.generate_content(
        model=MODEL,
        contents=history,
        config=config,
    )
    history.append(reply.candidates[0].content)

    pollen_request = PollenRequest.model_validate_json(reply.text)

    if (
        pollen_request.question
        or pollen_request.today_date is None
        or pollen_request.pollen_type is None
    ):
        return pollen_request, None

    response = requests.post(
        f"{SERVICE}/predict",
        json={
            "today_date": pollen_request.today_date,
            "pollen_type": pollen_request.pollen_type,
        },
        timeout=10,
    )
    response.raise_for_status()

    return pollen_request, response.json()


if __name__ == "__main__":
    print("Skriv inn dato og pollentype, for eksempel «Bjørk 12.04.2025».")
    print("Trykk Enter på en tom linje for å avslutte.\n")

    while True:
        message = input("du: ")

        if not message.strip():
            break

        pollen_request, answer = get_forecast(message)

        if answer is None:
            print("bot:", pollen_request.question or "Jeg trenger litt mer informasjon.")
        else:
            forecast = answer["forecast"]
            print("bot: Prediksjoner:")

            for date, level in zip(
                forecast["Dato"],
                forecast["Pollen-nivå"],
            ):
                print(f"  {date}: {level}")

        print()