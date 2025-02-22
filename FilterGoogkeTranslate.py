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
        pattern = re.compile(
            r'([\u4e00-\u9fff]+)|'      # Tiếng Trung
            r'([\uac00-\ud7a3]+)|'      # Tiếng Hàn
            r'([\u0600-\u06ff\u0750-\u077f\u08a0-\u08ff\ufb50-\ufdff\ufe70-\ufeff]+)|'  # Tiếng Ả Rập
            r'([\u0e00-\u0e7f]+)|'      # Tiếng Thái
            r'([\u0400-\u04FF]+)|'      # Tiếng Nga
            r'([\u0370-\u03FF]+)|'      # Tiếng Hy Lạp
            r'([\u0530-\u058F]+)|'      # Tiếng Armenia
            r'([\u0590-\u05FF]+)|'      # Tiếng Hebrew
            r'([\u0900-\u097F]+)|'      # Tiếng Hindi
            r'([\u0980-\u09FF]+)|'      # Tiếng Bengali
            r'([\u0A00-\u0A7F]+)|'      # Tiếng Punjabi
            r'([\u0A80-\u0AFF]+)|'      # Tiếng Gujarati
            r'([\u0B00-\u0B7F]+)|'      # Tiếng Oriya
            r'([\u0B80-\u0BFF]+)|'      # Tiếng Tamil
            r'([\u0C00-\u0C7F]+)|'      # Tiếng Telugu
            r'([\u0C80-\u0CFF]+)|'      # Tiếng Kannada
            r'([\u0D00-\u0D7F]+)|'      # Tiếng Malayalam
            r'([\u0D80-\u0DFF]+)|'      # Tiếng Sinhala
            r'([\u0E80-\u0EFF]+)|'      # Tiếng Lao
            r'([\u0F00-\u0FFF]+)|'      # Tiếng Tạng
            r'([\u1000-\u109F]+)|'      # Tiếng Myanmar
            r'([\u1100-\u11FF]+)|'      # Tiếng Hangul Jamo
            r'([\u1200-\u137F]+)|'      # Tiếng Ethiopia
            r'([\u13A0-\u13FF]+)|'      # Tiếng Cherokee
            r'([\u1400-\u167F]+)|'      # Tiếng Canada bản địa
            r'([\u1680-\u169F]+)|'      # Tiếng Ogham
            r'([\u16A0-\u16FF]+)|'      # Tiếng Runic
            r'([\u1700-\u171F]+)|'      # Tiếng Tagalog
            r'([\u1720-\u173F]+)|'      # Tiếng Hanunoo
            r'([\u1740-\u175F]+)|'      # Tiếng Buhid
            r'([\u1760-\u177F]+)|'      # Tiếng Tagbanwa
            r'([\u1780-\u17FF]+)|'      # Tiếng Khmer
            r'([\u1800-\u18AF]+)|'      # Tiếng Mông Cổ
            r'([\u1E00-\u1EFF]+)|'      # Tiếng Latin mở rộng bổ sung
            r'([\u1F00-\u1FFF]+)|'      # Tiếng Hy Lạp mở rộng
            r'([\u2000-\u206F]+)|'      # Các ký hiệu chung
            r'([\u2070-\u209F]+)|'      # Các chỉ số và chỉ số phụ
            r'([\u20A0-\u20CF]+)|'      # Các ký hiệu tiền tệ
            r'([\u20D0-\u20FF]+)|'      # Các ký hiệu kết hợp
            r'([\u2100-\u214F]+)|'      # Các ký hiệu chữ cái
            r'([\u2150-\u218F]+)|'      # Các số dạng chữ cái
            r'([\u2190-\u21FF]+)|'      # Các ký hiệu mũi tên
            r'([\u2200-\u22FF]+)|'      # Các ký hiệu toán học
            r'([\u2300-\u23FF]+)|'      # Các ký hiệu kỹ thuật
            r'([\u2400-\u243F]+)|'      # Các ký hiệu điều khiển
            r'([\u2440-\u245F]+)|'      # Các ký hiệu quang học
            r'([\u2460-\u24FF]+)|'      # Các số dạng vòng
            r'([\u2500-\u257F]+)|'      # Các ký hiệu hộp
            r'([\u2580-\u259F]+)|'      # Các ký hiệu khối
            r'([\u25A0-\u25FF]+)|'      # Các ký hiệu hình học
            r'([\u2600-\u26FF]+)|'      # Các ký hiệu đa dạng
            r'([\u2700-\u27BF]+)|'      # Các ký hiệu dingbat
            r'([\u27C0-\u27EF]+)|'      # Các ký hiệu toán học bổ sung
            r'([\u27F0-\u27FF]+)|'      # Các ký hiệu mũi tên bổ sung
            r'([\u2800-\u28FF]+)|'      # Các ký hiệu Braille
            r'([\u2900-\u297F]+)|'      # Các ký hiệu mũi tên bổ sung-B
            r'([\u2980-\u29FF]+)|'      # Các ký hiệu toán học bổ sung-B
            r'([\u2A00-\u2AFF]+)|'      # Các ký hiệu toán học bổ sung-C
            r'([\u2B00-\u2BFF]+)|'      # Các ký hiệu mũi tên bổ sung-C
            r'([\u2C00-\u2C5F]+)|'      # Các ký hiệu Glagolitic
            r'([\u2C60-\u2C7F]+)|'      # Các ký hiệu Latin mở rộng-C
            r'([\u2C80-\u2CFF]+)|'      # Các ký hiệu Coptic
            r'([\u2D00-\u2D2F]+)|'      # Các ký hiệu Georgian bổ sung
            r'([\u2D30-\u2D7F]+)|'      # Các ký hiệu Tifinagh
            r'([\u2D80-\u2DDF]+)|'      # Các ký hiệu Ethiopia bổ sung
            r'([\u2DE0-\u2DFF]+)|'      # Các ký hiệu Cyrillic mở rộng-A
            r'([\u2E00-\u2E7F]+)|'      # Các ký hiệu bổ sung
            r'([\u2E80-\u2EFF]+)|'      # Các ký hiệu CJK bổ sung
            r'([\u2F00-\u2FDF]+)|'      # Các ký hiệu Radicals Kangxi
            r'([\u2FF0-\u2FFF]+)|'      # Các ký hiệu Ideographic mô tả
            r'([\u3000-\u303F]+)|'      # Các ký hiệu CJK
            r'([\u3040-\u309F]+)|'      # Các ký hiệu Hiragana
            r'([\u30A0-\u30FF]+)|'      # Các ký hiệu Katakana
            r'([\u3100-\u312F]+)|'      # Các ký hiệu Bopomofo
            r'([\u3130-\u318F]+)|'      # Các ký hiệu Hangul Compatibility Jamo
            r'([\u3190-\u319F]+)|'      # Các ký hiệu Kanbun
            r'([\u31A0-\u31BF]+)|'      # Các ký hiệu Bopomofo mở rộng
            r'([\u31C0-\u31EF]+)|'      # Các ký hiệu CJK Strokes
            r'([\u31F0-\u31FF]+)|'      # Các ký hiệu Katakana Phonetic Extensions
            r'([\u3200-\u32FF]+)|'      # Các ký hiệu Enclosed CJK Letters and Months
            r'([\u3300-\u33FF]+)|'      # Các ký hiệu CJK Compatibility
            r'([\u3400-\u4DBF]+)|'      # Các ký hiệu CJK Unified Ideographs Extension A
            r'([\u4DC0-\u4DFF]+)|'      # Các ký hiệu Yijing Hexagram
            r'([\u4E00-\u9FFF]+)|'      # Các ký hiệu CJK Unified Ideographs
            r'([\uA000-\uA48F]+)|'      # Các ký hiệu Yi Syllables
            r'([\uA490-\uA4CF]+)|'      # Các ký hiệu Yi Radicals
            r'([\uA4D0-\uA4FF]+)|'      # Các ký hiệu Lisu
            r'([\uA500-\uA63F]+)|'      # Các ký hiệu Vai
            r'([\uA640-\uA69F]+)|'      # Các ký hiệu Cyrillic mở rộng-B
            r'([\uA6A0-\uA6FF]+)|'      # Các ký hiệu Bamum
            r'([\uA700-\uA71F]+)|'      # Các ký hiệu Modifier Tone Letters
            r'([\uA720-\uA7FF]+)|'      # Các ký hiệu Latin mở rộng-D
            r'([\uA800-\uA82F]+)|'      # Các ký hiệu Syloti Nagri
            r'([\uA830-\uA83F]+)|'      # Các ký hiệu Common Indic Number Forms
            r'([\uA840-\uA87F]+)|'      # Các ký hiệu Phags-pa
            r'([\uA880-\uA8DF]+)|'      # Các ký hiệu Saurashtra
            r'([\uA8E0-\uA8FF]+)|'      # Các ký hiệu Devanagari mở rộng
            r'([\uA900-\uA92F]+)|'      # Các ký hiệu Kayah Li
            r'([\uA930-\uA95F]+)|'      # Các ký hiệu Rejang
            r'([\uA960-\uA97F]+)|'      # Các ký hiệu Hangul Jamo mở rộng-A
            r'([\uA980-\uA9DF]+)|'      # Các ký hiệu Javanese
            r'([\uA9E0-\uA9FF]+)|'      # Các ký hiệu Myanmar mở rộng-B
            r'([\uAA00-\uAA5F]+)|'      # Các ký hiệu Cham
            r'([\uAA60-\uAA7F]+)|'      # Các ký hiệu Myanmar mở rộng-A
            r'([\uAA80-\uAADF]+)|'      # Các ký hiệu Tai Viet
            r'([\uAAE0-\uAAFF]+)|'      # Các ký hiệu Meetei Mayek mở rộng
            r'([\uAB00-\uAB2F]+)|'      # Các ký hiệu Ethiopic mở rộng-A
            r'([\uAB30-\uAB6F]+)|'      # Các ký hiệu Latin mở rộng-E
            r'([\uAB70-\uABBF]+)|'      # Các ký hiệu Cherokee bổ sung
            r'([\uABC0-\uABFF]+)|'      # Các ký hiệu Meetei Mayek
            r'([\uAC00-\uD7AF]+)|'      # Các ký hiệu Hangul Syllables
            r'([\uD7B0-\uD7FF]+)|'      # Các ký hiệu Hangul Jamo mở rộng-B
            r'([\uD800-\uDB7F]+)|'      # Các ký hiệu High Surrogates
            r'([\uDB80-\uDBFF]+)|'      # Các ký hiệu High Private Use Surrogates
            r'([\uDC00-\uDFFF]+)|'      # Các ký hiệu Low Surrogates
            r'([\uE000-\uF8FF]+)|'      # Các ký hiệu Private Use Area
            r'([\uF900-\uFAFF]+)|'      # Các ký hiệu CJK Compatibility Ideographs
            r'([\uFB00-\uFB4F]+)|'      # Các ký hiệu Alphabetic Presentation Forms
            r'([\uFB50-\uFDFF]+)|'      # Các ký hiệu Arabic Presentation Forms-A
            r'([\uFE00-\uFE0F]+)|'      # Các ký hiệu Variation Selectors
            r'([\uFE10-\uFE1F]+)|'      # Các ký hiệu Vertical Forms
            r'([\uFE20-\uFE2F]+)|'      # Các ký hiệu Combining Half Marks
            r'([\uFE30-\uFE4F]+)|'      # Các ký hiệu CJK Compatibility Forms
            r'([\uFE50-\uFE6F]+)|'      # Các ký hiệu Small Form Variants
            r'([\uFE70-\uFEFF]+)|'      # Các ký hiệu Arabic Presentation Forms-B
            r'([\uFF00-\uFFEF]+)|'      # Các ký hiệu Halfwidth and Fullwidth Forms
            r'([\uFFF0-\uFFFF]+)'       # Các ký hiệu Specials
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
