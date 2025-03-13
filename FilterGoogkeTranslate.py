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
    description = ("Lọc các ký tự của các ngôn ngữ được hỗ trợ bởi mô hình Qwen2 "
                   "trong văn bản, sau đó dịch chúng sang tiếng Việt.")
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateMultiLanguageInputComponent"

    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info=("Nhập văn bản có thể chứa các ngôn ngữ khác nhau. "
                  "Các phần ký tự của các ngôn ngữ được hỗ trợ bởi Qwen2 sẽ được dịch sang tiếng Việt."),
            value="This is a test: 你好, 호텔, الفندق, สวัสดี, Привет, שלום, こんにちは, สวัสดี, မင်္ဂလာပါ, ជំរាបសួរ, नमस्ते, হ্যালো, world!",
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
        Lọc và dịch các ký tự của các ngôn ngữ được hỗ trợ bởi Qwen2 sang tiếng Việt.
        Chỉ dịch phần sau "</think>".
        Các ngôn ngữ bao gồm:
          - Tiếng Trung: [\u4e00-\u9fff]
          - Tiếng Hàn: [\uac00-\ud7a3]
          - Tiếng Ả Rập/Persian/Urdu: [\u0600-\u06ff, \u0750-\u077f, \u08a0-\u08ff, \ufb50-\ufdff, \ufe70-\ufeff]
          - Tiếng Thái: [\u0e00-\u0e7f]
          - Tiếng Cyrillic (ví dụ: tiếng Nga): [\u0400-\u04FF]
          - Tiếng Hebrew: [\u0590-\u05FF]
          - Tiếng Nhật (Hiragana): [\u3040-\u309F]
          - Tiếng Nhật (Katakana): [\u30A0-\u30FF]
          - Tiếng Lao: [\u0E80-\u0EFF]
          - Tiếng Burmese: [\u1000-\u109F]
          - Tiếng Khmer: [\u1780-\u17FF]
          - Tiếng Devanagari (Hindi): [\u0900-\u097F]
          - Tiếng Bengali: [\u0980-\u09FF]
        """
        # Split text based on "</think>" marker
        parts = text.split("</think>", 1)
        
        # If "</think>" is not found or it's the last part, process the whole text
        if len(parts) == 1:
            text_to_translate = text
            prefix = ""
        else:
            # Keep the first part (including "</think>") unchanged
            prefix = parts[0] + "</think>"
            # Only translate the second part
            text_to_translate = parts[1]
        
        pattern = re.compile(
            r'([\u4e00-\u9fff]+)|'        # Tiếng Trung
            r'([\uac00-\ud7a3]+)|'        # Tiếng Hàn
            r'([\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+)|'  # Tiếng Ả Rập/Persian/Urdu
            r'([\u0e00-\u0e7f]+)|'        # Tiếng Thái
            r'([\u0400-\u04FF]+)|'        # Cyrillic (ví dụ: tiếng Nga)
            r'([\u0590-\u05FF]+)|'        # Tiếng Hebrew
            r'([\u3040-\u309F]+)|'        # Hiragana (Nhật)
            r'([\u30A0-\u30FF]+)|'        # Katakana (Nhật)
            r'([\u0E80-\u0EFF]+)|'        # Tiếng Lao
            r'([\u1000-\u109F]+)|'        # Tiếng Burmese
            r'([\u1780-\u17FF]+)|'        # Tiếng Khmer
            r'([\u0900-\u097F]+)|'        # Tiếng Devanagari (Hindi)
            r'([\u0980-\u09FF]+)'         # Tiếng Bengali
        )
    
        async def translate_match(match):
            original = match.group(0)
            try:
                translated_result = await translator.translate(original, dest='vi')
                return match.start(), match.end(), translated_result.text
            except Exception as e:
                print(f"Translation error: {e}")
                return match.start(), match.end(), original
    
        async def replace_with_translation(text):
            tasks = [translate_match(match) for match in pattern.finditer(text)]
            translated_parts = await asyncio.gather(*tasks)
    
            # Sắp xếp kết quả theo vị trí xuất hiện trong văn bản
            translated_parts.sort(key=lambda x: x[0])
    
            result_text = ""
            last_pos = 0
            for start_pos, end_pos, translated_text in translated_parts:
                # Thêm phần văn bản chưa được dịch
                result_text += text[last_pos:start_pos]
                # Nếu cần, thêm khoảng trắng giữa các từ
                if result_text and result_text[-1].isalpha() and translated_text and translated_text[0].isalpha():
                    result_text += " " + translated_text.lower()
                else:
                    result_text += translated_text
                last_pos = end_pos
            result_text += text[last_pos:]  # Thêm phần còn lại của văn bản
            return result_text
    
        # Only translate the part after "</think>"
        translated_part = await replace_with_translation(text_to_translate)
        
        # Combine the unchanged prefix with the translated part
        return prefix + translated_part

    async def run_filter_and_translate(self) -> Message:
        input_text = self.input_text
        try:
            async with self.get_translator() as translator:
                final_text = await self._filter_and_translate_text(input_text, translator)
                return Message(text=final_text, sender="AI", sender_name="Chatbot")
        except Exception as e:
            return Message(text=f"Translation error: {e}", sender="AI", sender_name="Chatbot")
