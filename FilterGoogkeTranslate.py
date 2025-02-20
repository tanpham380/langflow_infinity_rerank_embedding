from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message
import re
import asyncio
from googletrans import Translator
from typing import AsyncIterator, Union, Generator
from contextlib import asynccontextmanager

class FilterAndTranslateComponent(Component):
    display_name = "Filter and Translate (Multi-Language Input)"
    description = "Filters Chinese, Korean, and Arabic characters in text and translates them to Vietnamese. Supports both full text and processes as a single Message output."
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateMultiLanguageInputComponent"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="Enter text that may contain Chinese, Korean, or Arabic characters. Can be full text or streaming chunks.",
            value="This is a test: 你好, 호텔, الفندق, world!",
            tool_mode=True,
        ),
    ]

    outputs = [
        Output(display_name="Translated Text", name="output", method="run_filter_and_translate")
    ]

    @asynccontextmanager
    async def get_translator(self):
        async with Translator() as translator:
            yield translator

    async def _filter_and_translate_text(self, text: str, translator: Translator) -> str:
        """
        Filters and translates Chinese, Korean, and Arabic characters in text.
        """
        # Regex to include Chinese, Korean, and Arabic characters
        # Chinese (U+4E00-U+9FFF), Hangul Syllables (U+AC00-U+D7A3), Arabic (U+0600-U+06FF) and Arabic Supplement (U+0750-U+077F, U+08A0-U+08FF, U+FB50-U+FDFF, U+FE70-U+FEFF)
        pattern = re.compile(r'([\u4e00-\u9fff]+)|([\uac00-\ud7a3]+)|([\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+)')

        async def translate_match(match):
            original = match.group(0)
            try:
                translated_result = await translator.translate(original, dest='vi')
                translated_text = translated_result.text
                pos = match.start()
                if pos > 0:
                    if text[pos - 1].isspace():
                        translated_text = translated_text[0].lower() + translated_text[1:]
                    else:
                        translated_text = " " + (translated_text[0].lower() + translated_text[1:])
                return translated_text
            except Exception as e:
                print(f"Translation error: {e}")
                return original

        async def replace_with_translation(text):
            result_text_parts = []
            last_pos = 0
            for match in pattern.finditer(text):
                start, end = match.start(), match.end()
                result_text_parts.append(text[last_pos:start])
                translated_text = await translate_match(match)
                result_text_parts.append(translated_text)
                last_pos = end
            result_text_parts.append(text[last_pos:])
            return "".join(result_text_parts)

        return await replace_with_translation(text)

    async def run_filter_and_translate(self) -> Message:
        """
        Processes input text and returns translated output as a single Message.
        """
        input_text = self.input_text
        final_text = ""

        try:
            async with self.get_translator() as translator:
                if isinstance(input_text, str):
                    # Handle full text input
                    final_text = await self._filter_and_translate_text(input_text, translator)

                elif isinstance(input_text, AsyncIterator):
                    # Handle streaming input
                    async for chunk in input_text:
                        if not chunk.strip():
                            continue
                        translated_chunk = await self._filter_and_translate_text(chunk, translator)
                        final_text += translated_chunk

                elif isinstance(input_text, Generator):
                    # Handle sync generator
                    for chunk in input_text:
                        if not chunk.strip():
                            continue
                        translated_chunk = await self._filter_and_translate_text(chunk, translator)
                        final_text += translated_chunk

                else:
                    return Message(
                        text="Unsupported input type. Please provide text or a text stream.",
                        sender="AI",
                        sender_name="Chatbot"
                    )

                return Message(text=final_text, sender="AI", sender_name="Chatbot")

        except Exception as e:
            return Message(
                text=f"An error occurred during translation: {str(e)}",
                sender="AI",
                sender_name="Chatbot"
            )