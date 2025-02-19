from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message
from langflow.inputs import BoolInput
import re
from googletrans import Translator
from typing import Union, Generator
from concurrent.futures import ThreadPoolExecutor
import asyncio  # New import

class FilterAndTranslateComponent(Component):
    display_name = "Filter and Translate"
    description = "Filters Chinese characters in text and translates them to Vietnamese. Supports both text and streaming inputs."
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateComponent"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="Enter text that may contain Chinese characters.",
            value="This is a test: 你好, world!",
            tool_mode=True,
        )
    ]

    outputs = [
        Output(display_name="Translated Text", name="output", method="run")
    ]

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.executor = ThreadPoolExecutor()  # Use the directly imported ThreadPoolExecutor

    def _filter_and_translate(self, text: str) -> str:
        """
        Filters Chinese characters and translates them to Vietnamese.
        """
        pattern = re.compile(r'([\u4e00-\u9fff]+)')
        translator = Translator()

        def translate_match(match):
            original = match.group(0)
            try:
                # Wrap the coroutine in asyncio.run() within the thread pool
                future = self.executor.submit(
                    lambda: asyncio.run(translator.translate(original, dest='vi'))
                )
                translated_result = future.result(timeout=10)  # Wait for result with timeout
                translated_text = translated_result.text
                return " " + translated_text  # Add space for separation
            except Exception as e:
                self.log(f"Translation error: {e}")
                return original  # Return original text in case of error

        result_text = pattern.sub(translate_match, text)
        return result_text.strip()  # remove leading/trailing spaces

    def run(self) -> Union[Message, Generator[Message, None, None]]:
        """
        Processes input text and handles both streaming and non-streaming inputs.
        """
        self.log(f"Input type in run: {type(self.input_text)}")  # Debug print

        if hasattr(self.input_text, '__iter__') and not isinstance(self.input_text, str):
            self.log("Streaming input detected")  # Debug print
            def stream_translate():
                for i, chunk in enumerate(self.input_text):
                    self.log(f"Processing chunk {i}: {chunk}")  # Debug print
                    translated_chunk = self._filter_and_translate(chunk)
                    yield Message(text=translated_chunk, sender="AI", sender_name="Chatbot")
                self.log("Finished streaming input")  # Debug print
            return stream_translate()  # Return the generator
        else:
            self.log("Non-streaming input detected")  # Debug print
            final_text = self._filter_and_translate(self.input_text)
            return Message(text=final_text, sender="AI", sender_name="Chatbot")
