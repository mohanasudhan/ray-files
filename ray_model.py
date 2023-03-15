from starlette.requests import Request

# import ray libraries and serve component
import ray
from ray import serve

from transformers import pipeline

#annotate the model class - Actor (stateful)
@serve.deployment
class Translator:
    def __init__(self):
        # Load model
        self.model = pipeline("translation_en_to_fr", model="t5-small")

    # Pre and post processing
    def translate(self, text: str) -> str:
        # Run inference
        model_output = self.model(text)
        # Post-process output to return only the translation text
        translation = model_output[0]["translation_text"]
        return translation

    # predict handler (http handler)
    async def __call__(self, http_request: Request) -> str:
        english_text: str = await http_request.json()
        return self.translate(english_text)


translator = Translator.bind()
