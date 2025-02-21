from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message
import re
import asyncio
from googletrans import Translator
from typing import AsyncIterator, Union, Generator
from contextlib import asynccontextmanager

class FilterAndTranslateComponent(Component):
    display_name = "Filter and Translate (Multi-Language Input with Thai)"
    description = ("Lọc các ký tự của các ngôn ngữ Trung, Hàn, Ả Rập và Thái "
                   "trong văn bản, sau đó dịch chúng sang tiếng Việt. Hỗ trợ xử lý văn bản đầy đủ hoặc streaming.")
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateMultiLanguageInputComponent"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info=("Nhập văn bản có thể chứa tiếng Trung, Hàn, Ả Rập và Thái. "
                  "Các phần ký tự của các ngôn ngữ này sẽ được dịch sang tiếng Việt."),
            value="This is a test: 你好, 호텔, الفندق, สวัสดี, world!",
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
        Lọc và dịch các ký tự của các ngôn ngữ Trung, Hàn, Ả Rập và Thái sang tiếng Việt.
        """
        # Regex đã được cập nhật bao gồm tiếng Thái (U+0E00-U+0E7F)
        pattern = re.compile(
            r'([\u4e00-\u9fff]+)|'      # Tiếng Trung
            r'([\uac00-\ud7a3]+)|'      # Tiếng Hàn
            r'([\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+)|'  # Tiếng Ả Rập
            r'([\u0e00-\u0e7f]+)'       # Tiếng Thái
        )

        async def translate_match(match):
            original = match.group(0)
            try:
                translated_result = await translator.translate(original, dest='vi')
                translated_text = translated_result.text
                pos = match.start()
                # Giữ nguyên khoảng trắng hoặc thay đổi chữ cái đầu nếu cần
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
        Xử lý đầu vào và trả về văn bản đã được dịch (các phần ký tự của Trung, Hàn, Ả Rập và Thái được dịch sang tiếng Việt)
        dưới dạng một Message.
        """
        input_text = self.input_text
        final_text = ""

        try:
            async with self.get_translator() as translator:
                if isinstance(input_text, str):
                    # Xử lý văn bản đầy đủ
                    final_text = await self._filter_and_translate_text(input_text, translator)

                elif isinstance(input_text, AsyncIterator):
                    # Xử lý streaming input (AsyncIterator)
                    async for chunk in input_text:
                        if not chunk.strip():
                            continue
                        translated_chunk = await self._filter_and_translate_text(chunk, translator)
                        final_text += translated_chunk

                elif isinstance(input_text, Generator):
                    # Xử lý streaming input (sync Generator)
                    for chunk in input_text:
                        if not chunk.strip():
                            continue
                        translated_chunk = await self._filter_and_translate_text(chunk, translator)
                        final_text += translated_chunk

                else:
                    return Message(
                        text="Unsupported input type. Vui lòng cung cấp văn bản hoặc stream văn bản.",
                        sender="AI",
                        sender_name="Chatbot"
                    )

                return Message(text=final_text, sender="AI", sender_name="Chatbot")

        except Exception as e:
            return Message(
                text=f"Đã xảy ra lỗi khi dịch: {str(e)}",
                sender="AI",
                sender_name="Chatbot"
            )
