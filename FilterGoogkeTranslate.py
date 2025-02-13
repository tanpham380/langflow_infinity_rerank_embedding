from langflow.custom import Component
from langflow.io import MessageTextInput, Output
from langflow.schema import Message  # Import the Message object
from langflow.inputs import BoolInput  # import Boolean input type
import re
import asyncio
from googletrans import Translator

class FilterAndTranslateComponent(Component):
    display_name = "Filter and Translate"
    description = "Filters Chinese characters in text and translates them to Vietnamese."
    documentation = "https://py-googletrans.readthedocs.io/en/latest/"
    icon = "google"
    name = "FilterAndTranslateComponent"

    # The input can be a full text string or a generator of text chunks.
    inputs = [
        MessageTextInput(
            name="input_text",
            display_name="Input Text",
            info="Enter text that may contain Chinese characters. This can be a full string or a generator yielding text chunks.",
            value="This is a test: 你好, world!",
            tool_mode=True,
        ),
        BoolInput(
            name="openai_stream",
            display_name="Stream Output",
            info="If true, the translated text will be processed in a streaming fashion but returned as a single Message.",
            value=False,
        )
    ]

    outputs = [
        Output(display_name="Translated Text", name="output", method="run_filter_and_translate")
    ]

    def _filter_and_translate(self, text: str) -> str:
        """
        Thay thế các ký tự tiếng Trung trong text bằng bản dịch tiếng Việt.
        """
        # Tạo pattern để tìm các ký tự tiếng Trung
        pattern = re.compile(r'([\u4e00-\u9fff]+)')
    
        # Tạo và thiết lập event loop duy nhất cho toàn bộ quá trình dịch
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        
        # Tạo đối tượng Translator sau khi đã thiết lập event loop
        translator = Translator()
    
        # Hàm nội bộ dùng cho mỗi match trong regex
        def translate_match(match):
            original = match.group(0)
            # Sử dụng event loop đã tạo để chạy asynchronous translation
            translated_result = loop.run_until_complete(translator.translate(original, dest='vi'))
            translated_text = translated_result.text
            pos = match.start()
            # Điều chỉnh chữ cái đầu của kết quả dịch nếu cần
            if pos > 0:
                if text[pos - 1].isspace():
                    translated_text = translated_text[0].lower() + translated_text[1:]
                else:
                    translated_text = " " + (translated_text[0].lower() + translated_text[1:])
            return translated_text
    
        # Thực hiện thay thế cho toàn bộ text
        result_text = pattern.sub(translate_match, text)
    
        # Đóng event loop sau khi hoàn tất
        loop.close()
        return result_text

    def run_filter_and_translate(self) -> Message:
        """
        Processes the input text. If 'openai_stream' is True, processes the input in chunks and then combines the results;
        otherwise, processes the entire text at once.
        Returns a single Message object.
        """
        if self.openai_stream:
            # If streaming is enabled, check if input_text is iterable (i.e. already chunked)
            if hasattr(self.input_text, '__iter__') and not isinstance(self.input_text, str):
                chunks = [self._filter_and_translate(chunk) for chunk in self.input_text]
            else:
                # Otherwise, split the text into words as a simple chunking method.
                words = self.input_text.split()
                chunks = [self._filter_and_translate(word) + " " for word in words]
            final_text = "".join(chunks)
        else:
            final_text = self._filter_and_translate(self.input_text)
        return Message(text=final_text, sender="AI", sender_name="Chatbot")
