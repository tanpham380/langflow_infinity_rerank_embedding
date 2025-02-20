import re
from pathlib import Path
from typing import Optional, List, Tuple
from langflow.custom import Component
from langflow.io import BoolInput, FileInput, Output, MultilineInput, DataInput
from langflow.schema import Data
from docx import Document

def normalize_line(line: str) -> str:
    """
    Loại bỏ số thứ tự ở đầu dòng và khoảng trắng thừa.
    """
    return re.sub(r'^\d+\.\s*', '', line).strip()

def clean_keywords(keywords: List[str]) -> List[str]:
    """
    Remove specified keywords and clean up the keyword list.
    """
    excluded_keywords = {
        'bảo lưu', 'chi phí', 'chính sách', 'giá',
        'hoàn tiền', 'học phí', 'ưu đãi' , 'yêu cầu đầu vào', 'ưu đãi học phí'
    }

    cleaned = []
    for kw in keywords:
        kw = kw.lower().strip()
        if kw and kw not in excluded_keywords:
            cleaned.append(kw)

    return sorted(list(set(cleaned)))

class ListFileDocxComponent(Component):
    display_name = "List File Docx Component"
    description = "Load and process DOCX files with semantic section splitting. Supports flexible program intro detection."
    icon = "File"
    name = "ListFileDocxComponent"

    inputs = [
        FileInput(
            name="path",
            display_name="File Path",
            file_types=["docx"],
            info="Upload a DOCX file to process."
        ),
        MultilineInput(
            name="meta_keywords",
            display_name="Meta Keywords (comma-separated)",
            info="Meta keywords to append to chunks.",
            value="marketing,course,hotel"
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception.",
            value=False # Set default value to False
        ),
        DataInput(
            name="data",
            display_name="Data List",
            info="Input list of Data objects from DirectoryComponent.",
        ),
    ]

    outputs = [Output(display_name="Data Chunks", name="chunks", method="process_docx")]

    # Cấu trúc chunk theo định dạng file tài liệu - CÓ CẤU HÌNH INTRO CHUNK
    CHUNK_STRUCTURE = {
        "general_info": ["Thông Tin Tổng Quan", "Thông tin chung"],
        "tuition": ["Học phí", "Chi phí"],
        "core_values": ["Giá trị cốt lõi", "Giá trị nhận được"],
        "job_support": ["Chính sách giới thiệu việc làm", "Hỗ trợ việc làm"],
        "course_content": ["Nội dung khóa học", "Chương trình học"],
        "entry_requirements": ["Yêu Cầu Đầu Vào", "Điều kiện nhập học"],
        "support_policy": ["Chính sách hỗ trợ", "Hỗ trợ học viên"],
        "dynamic_data": ["Dữ liệu động", "Thông tin khác"],
        "type_of_training": ["Hình thức học", "Hình thức đào tạo"],
        "program_intro_config": { # Cấu hình cho chunk giới thiệu chương trình
            "chunk_type": "program", # Loại chunk chứa phần giới thiệu (luôn là 'program')
            "intro_phrase_patterns": [ # Danh sách các regex patterns để nhận diện phần giới thiệu
                r"Hiện tại Hướng Nghiệp Á Âu / CET đào tạo các khoá học của Nhà hàng Khách sạn:", # Pattern cũ cho NHKS
                r"Học viện Lập trình Mobile cung cấp nhiều chương trình đào tạo đa dạng,", # Pattern cho Mobile
                r"Học viện Du Lịch cung cấp các chương trình đào tạo đa dạng,", # Pattern cho Du Lịch
                r"Hiện tại .* đào tạo các khoá học về.*:", # Pattern tổng quát hơn
                r".*cung cấp các chương trình đào tạo đa dạng.*:", # Pattern tổng quát hơn nữa
                r"5\. Nội Dung Khóa Học Học viện Lập trình Mobile cung cấp nhiều chương trình đào tạo đa dạng,", # Ví dụ pattern cụ thể cho Mobile - có thể cần điều chỉnh
                r"5\. Nội Dung Khóa Học Học viện Du Lịch cung cấp các chương trình đào tạo đa dạng,", # Ví dụ pattern cụ thể cho Du Lịch - có thể cần điều chỉnh
            ]
        }
    }


    def extract_text_from_docx(self, file_path: Path) -> str:
        """
        Extract text content from a DOCX file.
        """
        silent_errors = self.silent_errors or False # Get runtime value, default to False if None
        try:
            doc = Document(file_path)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            self.log(f"Error extracting text from DOCX: {e}")
            if not silent_errors:
                raise
            return ""

    def _extract_course_name(self, text: str) -> str:
        """
        Extract course name from text content.
        """
        pattern = r"(?:•\s*)?Tên khóa học:\s*(.+)", re.IGNORECASE
        match = re.search(pattern, text)
        return match.group(1).strip() if match else ""


    def _create_semantic_chunks(self, text: str) -> List[Tuple[str, str]]:
        """
        Create semantic chunks based on the defined structure.
        Improved logic for chunking, more readable and efficient.
        """
        lines = text.strip().splitlines()
        chunks = []
        current_section: Optional[str] = None
        current_content: List[str] = []
        program_names_for_intro_chunk: List[str] = []

        def get_chunk_type(line: str) -> Optional[str]:
            normalized_line = normalize_line(line)
            for chunk_type, sections in self.CHUNK_STRUCTURE.items():
                # Bỏ qua 'program_intro_config' khi xác định chunk type thông thường
                if chunk_type == "program_intro_config":
                    continue
                for title in sections:
                    if re.fullmatch(rf"{re.escape(title)}\s*:?", normalized_line, re.IGNORECASE):
                        return chunk_type
            return None

        for line in lines:
            line = line.strip()
            if not line: # Skip empty lines
                if current_section: # But keep empty lines within a chunk if already started
                    current_content.append("")
                continue

            new_chunk_type = get_chunk_type(line)

            if new_chunk_type:
                if current_section and current_content:
                    chunk_text = "\n".join(current_content).strip()
                    if chunk_text:
                        if current_section == "course_content":
                            program_chunks, program_names = self._split_program_content(chunk_text)
                            chunks.extend(program_chunks)
                            program_names_for_intro_chunk.extend(program_names)
                        else:
                            chunks.append((current_section, chunk_text))

                current_section = new_chunk_type
                current_content = []
            elif current_section:
                current_content.append(line)

        # Add the last chunk if content remains
        if current_section and current_content:
            chunk_text = "\n".join(current_content).strip()
            if chunk_text:
                if current_section == "course_content":
                    program_chunks, program_names = self._split_program_content(chunk_text)
                    chunks.extend(program_chunks)
                    program_names_for_intro_chunk.extend(program_names)
                else:
                    chunks.append((current_section, chunk_text))

        # Update intro program chunk with program names - SỬ DỤNG CONFIG
        intro_config = self.CHUNK_STRUCTURE.get("program_intro_config")
        if intro_config and intro_config.get("chunk_type") == "program" and intro_config.get("intro_phrase_patterns"):
            intro_chunk_type = intro_config["chunk_type"]
            intro_phrase_patterns = intro_config["intro_phrase_patterns"]

            for i, (chunk_type, content) in enumerate(chunks):
                if chunk_type == intro_chunk_type:
                    for pattern in intro_phrase_patterns:
                        if re.search(pattern, content, re.IGNORECASE): # Tìm kiếm pattern
                            program_names_str = "\n".join([f"+{name}" for name in program_names_for_intro_chunk])
                            updated_content = re.sub(pattern, r"\g<0>\n" + program_names_str, content, flags=re.IGNORECASE) # Thêm program names sau intro phrase, giữ lại intro phrase gốc
                            chunks[i] = (chunk_type, updated_content)
                            break # Thoát khỏi vòng lặp patterns sau khi tìm thấy và cập nhật

        return chunks


    def _split_program_content(self, content: str) -> Tuple[List[Tuple[str, str]], List[str]]:
        """
        Split program content into chunks and extract program names.
        Improved logic for clarity and handling header info.
        """
        program_chunks: List[Tuple[str, str]] = []
        program_names: List[str] = []
        header_lines: List[str] = []
        current_program_content: List[str] = []
        keyword_line: Optional[str] = None
        in_header_section = True # More descriptive flag

        lines = content.strip().splitlines()

        for line in lines:
            line = line.strip()
            if not line:
                continue

            if in_header_section:
                if line.lower().startswith("từ khoá:"):
                    keyword_line = line
                    in_header_section = False # Header section ends after keyword line
                elif re.match(r"^Chương trình học \d+:", line, re.IGNORECASE):
                    in_header_section = False # Header section ends before program starts
                else:
                    header_lines.append(line)

            if not in_header_section:
                if line.lower().startswith("từ khoá:"): # Ignore keyword lines within program descriptions
                    continue

            program_match = re.match(r"^Chương trình học \d+:\s*(.+)$", line, re.IGNORECASE)
            if program_match:
                program_name = program_match.group(1).strip()
                program_names.append(program_name)
                if current_program_content:
                    program_text = "\n".join(current_program_content).strip()
                    if program_text:
                        program_chunks.append(("program", program_text))
                current_program_content = [line] # Start new program content with program title line
            else:
                current_program_content.append(line)

        # Add the last program content
        if current_program_content:
            program_text = "\n".join(current_program_content).strip()
            if program_text:
                program_chunks.append(("program", program_text))

        # Create header chunk if header lines or keyword line exist
        if header_lines or keyword_line:
            header_text_lines = header_lines[:] # Copy to avoid modifying original
            if keyword_line:
                header_text_lines.append(keyword_line)
            header_text = "\n".join(header_text_lines).strip()
            if header_text:
                program_chunks.insert(0, ("program", header_text)) # Insert header at the beginning


        return program_chunks, program_names


    def _process_keywords(self, file_path: Optional[Path] = None) -> List[str]:
        """
        Chỉ lấy các keyword liên quan tới tên file.
        """
        keywords = set()

        # Nếu có file, thêm tên file (không có phần mở rộng)
        if file_path:
            keywords.add(file_path.stem.lower())

        return list(keywords)

    def _format_metadata(self, chunk_id: str, chunk_type: str, source: str, keywords: List[str], program_names: Optional[List[str]] = None) -> dict:
        """
        Định dạng metadata với các tag chỉ liên quan tới tên file (keywords truyền vào) và program names.
        """
        tags = [f"#{kw}" for kw in keywords if kw]
        if chunk_type == "program" and program_names:
            tags.extend([f"#{pn.lower()}" for pn in program_names])

        return {
            "id": chunk_id,
            "type": chunk_type,
            "source": source,
            "tags": tags
        }

    def process_docx(self) -> List[Data]: # Loại bỏ các arguments ở đây
        """
        Processes a DOCX file to extract semantic chunks.

        Returns:
            A list of Data objects, where each Data object represents a semantic chunk
            with formatted content and metadata.
        """
        data_chunks = []
        silent_errors = self.silent_errors or False # Get runtime value, default to False if None

        self.log(f"process_docx called with path: {self.path}, data: {self.data}, silent_errors: {silent_errors}, meta_keywords: {self.meta_keywords}") # Logging with self

        if not self.path and not self.data: # Use 'not' for clarity and handle both None and empty lists
            self.log("No valid input provided for processing during component RUN.")
            return []

        try:
            # Handle DOCX file if provided
            if self.path:
                resolved_path = Path(self.resolve_path(self.path))
                if resolved_path.suffix.lower() != ".docx":
                    raise ValueError("Invalid file type. Only DOCX files are supported.")

                self.log(f"Processing DOCX file: {resolved_path.name}")
                text = self.extract_text_from_docx(resolved_path)
                chunks = self._create_semantic_chunks(text)
                keywords_list = self._process_keywords(resolved_path) # Get only filename as keyword

                # Add meta keywords if provided and not empty
                if self.meta_keywords:
                    meta_keywords_list = [kw.strip().lower() for kw in self.meta_keywords.split(',') if kw.strip()]
                    keywords_list.extend(meta_keywords_list)
                    keywords_list = sorted(list(set(keywords_list)))


                for chunk_id, (chunk_type, content) in enumerate(chunks):
                    program_names_in_chunk = None
                    if chunk_type == 'program':
                        program_names_in_chunk = []
                        # Extract program names again if needed for more granular tags,
                        # or rely on program names extracted earlier in _create_semantic_chunks
                        program_name_match = re.search(r"^Chương trình học \d+:\s*(.+)$", content, re.IGNORECASE)
                        if program_name_match:
                            program_names_in_chunk = [program_name_match.group(1).strip()]


                    metadata = self._format_metadata(
                        f"file_chunk_{chunk_id}",
                        chunk_type,
                        resolved_path.name,
                        keywords_list,
                        program_names_in_chunk # Pass program names for metadata
                    )

                    formatted_content = (
                        f"Type: {chunk_type}\n"
                        f"Content:\n{content}\n"
                        f"Metadata: {metadata}"
                    )
                    data_chunks.append(Data(text=formatted_content))

            # Handle Data objects if provided
            if self.data:
                self.log("Processing provided Data objects...")
                for data_idx, data_item in enumerate(self.data):
                    text = data_item.text
                    chunks = self._create_semantic_chunks(text)
                    keywords_list = self._process_keywords() # Keywords from Data input are based on content only

                    for chunk_id, (chunk_type, content) in enumerate(chunks):
                        metadata = self._format_metadata(
                            f"data_{data_idx}_chunk_{chunk_id}",
                            chunk_type,
                            "data input",
                            keywords_list,
                        )

                        formatted_content = (
                            f"Type: {chunk_type}\n"
                            f"Content:\n{content}\n"
                            f"Metadata: {metadata}"
                        )
                        data_chunks.append(Data(text=formatted_content))


            self.log(f"Successfully processed inputs into {len(data_chunks)} semantic chunks.")
            return data_chunks

        except Exception as e:
            self.log(f"Error processing input: {str(e)}")
            if not silent_errors:
                raise
            return []