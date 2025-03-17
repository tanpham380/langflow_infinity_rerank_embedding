from langflow.custom import Component
from langflow.io import MessageTextInput, Output, BoolInput
from langflow.schema import Message
import re
import asyncio
from googletrans import Translator

class FilterAndTranslateComponent(Component):
    display_name = "Filter and Translate (Multi-Language Input)"
    description = ("Lọc các ký tự của các ngôn ngữ được hỗ trợ bởi mô hình Qwen2 "
                   "trong văn bản, sau đó dịch chúng sang tiếng Việt.")
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateMultiLanguageInputComponent"

    pattern = re.compile(
        r'([\u4e00-\u9fff]+)|'        # Chinese
        r'([\uac00-\ud7a3]+)|'        # Korean
        r'([\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+)|'  # Arabic/Persian/Urdu
        r'([\u0e00-\u0e7f]+)|'        # Thai
        r'([\u0400-\u04FF]+)|'        # Cyrillic (e.g., Russian)
        r'([\u0590-\u05FF]+)|'        # Hebrew
        r'([\u3040-\u309F]+)|'        # Hiragana (Japanese)
        r'([\u30A0-\u30FF]+)|'        # Katakana (Japanese)
        r'([\u0E80-\u0EFF]+)|'        # Lao
        r'([\u1000-\u109F]+)|'        # Burmese
        r'([\u1780-\u17FF]+)|'        # Khmer
        r'([\u0900-\u097F]+)|'        # Devanagari (Hindi)
        r'([\u0980-\u09FF]+)'         # Bengali
    )

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info=("Nhập văn bản có thể chứa các ngôn ngữ khác nhau. "
                  "Các phần ký tự của các ngôn ngữ được hỗ trợ bởi Qwen2 sẽ được dịch sang tiếng Việt."),
            value="This is a test: 你好, 호텔, الفندق, สวัสดี, Привет, שלום, こんにちは, สวัสดี, မင်္ဂလာပါ, ជំរាបសួរ, नमस्ते, হ্যালো, world!",
            tool_mode=True,
        ),
        BoolInput(
            name="remove_think_tags",
            display_name="Remove <think> tags",
            info="Nếu tích chọn, các thẻ <think> và nội dung bên trong sẽ bị xóa hoàn toàn. Nếu không, giữ nguyên thẻ và nội dung bên trong, chỉ dịch văn bản bên ngoài.",
            value=False,
        ),
    ]

    outputs = [
        Output(display_name="Translated Text", name="output", method="run_filter_and_translate")
    ]

    def get_translator(self):
        """Initialize and return a Translator instance."""
        return Translator()

    async def translate_text(self, text: str, translator: Translator) -> str:
        """Translate foreign language segments in the text to Vietnamese."""
        async def translate_match(match):
            original = match.group(0)
            try:
                # Directly await the async method instead of using run_in_executor
                translated_result = await translator.translate(original, dest='vi')
                self.log(f"Translated '{original}' to '{translated_result.text}'")
                return match.start(), match.end(), translated_result.text
            except Exception as e:
                self.log(f"Translation error: {e}")
                print(f"Translation error: {e}")
                return match.start(), match.end(), original
    
        tasks = [translate_match(match) for match in self.pattern.finditer(text)]
        translated_parts = await asyncio.gather(*tasks)
        translated_parts.sort(key=lambda x: x[0])
        result_text = ""
        last_pos = 0
        for start_pos, end_pos, translated_text in translated_parts:
            result_text += text[last_pos:start_pos]
            if result_text and result_text[-1].isalpha() and translated_text and translated_text[0].isalpha():
                result_text += " " + translated_text.lower()
            else:
                result_text += translated_text
            last_pos = end_pos
        result_text += text[last_pos:]
        return result_text

    async def _filter_and_translate_text(self, text: str, translator: Translator, remove_think_tags: bool) -> str:
        """Filter and translate text, handling <think> tags based on remove_think_tags."""
        if remove_think_tags:
            text = re.sub(r'<think>.*?</think>', '', text, flags=re.DOTALL)
            return await self.translate_text(text, translator)
        else:
            parts = re.split(r'(<think>.*?</think>)', text, flags=re.DOTALL)
            processed_parts = []
            for i, part in enumerate(parts):
                if i % 2 == 0:  # Non-<think> part
                    translated_part = await self.translate_text(part, translator)
                    processed_parts.append(translated_part)
                else:  # <think> part
                    processed_parts.append(part)
            return ''.join(processed_parts)

    async def run_filter_and_translate(self) -> Message:
        """Run the filter and translate process."""
        input_text = self.input_text
        remove_think_tags = self.remove_think_tags
        try:
            translator = self.get_translator()
            final_text = await self._filter_and_translate_text(input_text, translator, remove_think_tags)
            return Message(text=final_text, sender="AI", sender_name="Chatbot")
        except Exception as e:
            return Message(text=f"Translation error: {e}", sender="AI", sender_name="Chatbot")