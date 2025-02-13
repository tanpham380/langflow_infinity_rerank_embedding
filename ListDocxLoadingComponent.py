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
    display_name = "ListFileDocxComponent"
    description = "Load and process DOCX files with semantic section splitting."
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
            value="marketing,sales"
        ),
        BoolInput(
            name="silent_errors",
            display_name="Silent Errors",
            advanced=True,
            info="If true, errors will not raise an exception."
        ),
        DataInput(
            name="data",
            display_name="Data List",
            info="Input list of Data objects from DirectoryComponent.",
        ),
    ]

    outputs = [Output(display_name="Data Chunks", name="chunks", method="process_docx")]

    # Cấu trúc chunk theo định dạng file tài liệu
    CHUNK_STRUCTURE = {
        "general_info": ["Thông Tin Tổng Quan"],
        "tuition": ["Học phí"],
        "core_values": ["Giá trị cốt lõi"],
        "job_support": ["Chính sách giới thiệu việc làm"],
        "course_content": ["Nội dung khóa học"],
        "entry_requirements": ["Yêu Cầu Đầu Vào"],
        "support_policy": ["Chính sách hỗ trợ"],
        "dynamic_data": ["Dữ liệu động"]
    }

    def extract_text_from_docx(self, file_path: Path) -> str:
        """
        Extract text content from a DOCX file.
        """
        try:
            doc = Document(file_path)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            self.log(f"Error extracting text from DOCX: {e}")
            if not self.silent_errors:
                raise
            return ""

    def _extract_course_name(self, text: str) -> str:
        """
        Extract course name from text content.
        """
        pattern = r"(?:•\s*)?Tên khóa học:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""

    def _create_semantic_chunks(self, text: str) -> List[Tuple[str, str]]:
        """
        Create semantic chunks based on the defined structure.
        """
        lines = text.splitlines()
        chunks = []
        current_section = None
        current_content = []
        
        def is_only_title(content: str, title: str) -> bool:
            content_clean = content.strip().rstrip(':').lower()
            title_clean = title.strip().rstrip(':').lower()
            return content_clean == title_clean
        
        def get_chunk_type(line: str) -> Tuple[Optional[str], Optional[str]]:
            normalized_line = normalize_line(line)
            for chunk_type, sections in self.CHUNK_STRUCTURE.items():
                if any(re.search(rf"^{re.escape(title)}", normalized_line, re.IGNORECASE) for title in sections):
                    return chunk_type, sections[0]
            return None, None

        for line in lines:
            new_chunk_type, section_title = get_chunk_type(line)
            normalized_line = normalize_line(line)
            if new_chunk_type:
                if current_section is None:
                    current_section = new_chunk_type
                    if normalized_line.rstrip(':').lower() == section_title.rstrip(':').lower():
                        current_content = []
                    else:
                        current_content = [line]
                else:
                    if new_chunk_type == current_section:
                        if normalized_line.rstrip(':').lower() != section_title.rstrip(':').lower():
                            current_content.append(line)
                    else:
                        chunk_text = "\n".join(current_content).strip()
                        if not is_only_title(chunk_text, self.CHUNK_STRUCTURE[current_section][0]):
                            if current_section == "course_content":
                                program_chunks = self._split_program_content(chunk_text)
                                chunks.extend(program_chunks)
                            else:
                                chunks.append((current_section, chunk_text))
                        current_section = new_chunk_type
                        if normalized_line.rstrip(':').lower() == section_title.rstrip(':').lower():
                            current_content = []
                        else:
                            current_content = [line]
            else:
                if current_section:
                    current_content.append(line)

        if current_section:
            chunk_text = "\n".join(current_content).strip()
            if not is_only_title(chunk_text, self.CHUNK_STRUCTURE[current_section][0]):
                if current_section == "course_content":
                    program_chunks = self._split_program_content(chunk_text)
                    chunks.extend(program_chunks)
                else:
                    chunks.append((current_section, chunk_text))
        
        return chunks

    def _split_program_content(self, content: str) -> List[Tuple[str, str]]:
        """
        Split program content into chunks.
        """
        lines = content.splitlines()
        header_lines = []
        keyword_line = ""
        program_start_idx = None
    
        for i, line in enumerate(lines):
            if re.match(r"^Chương trình học \d+:", line, re.IGNORECASE):
                program_start_idx = i
                break
            if line.strip().lower().startswith("từ khoá:"):
                keyword_line = line.strip()
            else:
                header_lines.append(line.strip())
    
        chunks = []
        if program_start_idx is not None:
            header_content = "\n".join(header_lines).strip()
    
            program_section_text = "\n".join(lines[program_start_idx:])
            program_names = re.findall(
                r"^Chương trình học \d+:\s*(.+)$",
                program_section_text,
                flags=re.IGNORECASE | re.MULTILINE
            )
            if program_names:
                header_content += "\n" + "\n".join("+" + name.strip() for name in program_names)
            if keyword_line:
                header_content += "\n" + keyword_line
    
            chunks.append(("program", header_content))
            current_chunk_lines = []
            for line in lines[program_start_idx:]:
                if re.match(r"^Chương trình học \d+:", line, re.IGNORECASE):
                    if current_chunk_lines:
                        program_content = "\n".join(current_chunk_lines).strip()
                        chunks.append(("program", program_content))
                        current_chunk_lines = []
                current_chunk_lines.append(line.strip())
            if current_chunk_lines:
                program_content = "\n".join(current_chunk_lines).strip()
                chunks.append(("program", program_content))
        else:
            header_content = "\n".join(header_lines).strip()
            if keyword_line:
                header_content += "\n" + keyword_line
            chunks.append(("program", header_content))
    
        return chunks

    def _process_keywords(self, text: str, file_path: Optional[Path] = None) -> List[str]:
        """
        Chỉ lấy các keyword liên quan tới tên khóa học và tên file.
        """
        keywords = set()
        
        # Nếu có file, thêm tên file (không có phần mở rộng)
        if file_path:
            keywords.add(file_path.stem.lower())
        
        # Lấy tên khóa học từ nội dung (sử dụng pattern của "Tên khóa học:")
        course_name = self._extract_course_name(text)
        if course_name:
            keywords.add(course_name.lower())
            
        return list(keywords)

    def _format_metadata(self, chunk_id: str, chunk_type: str, source: str, keywords: List[str]) -> dict:
        """
        Định dạng metadata với các tag chỉ liên quan tới tên khóa học và tên file.
        """
        return {
            "id": chunk_id,
            "type": chunk_type,
            "source": source,
            "tags": [f"#{kw}" for kw in keywords if kw]
        }

    def process_docx(self) -> List[Data]:
        data_chunks = []

        try:
            # Handle DOCX file if provided
            if self.path:
                resolved_path = Path(self.resolve_path(self.path))
                if resolved_path.suffix.lower() != ".docx":
                    raise ValueError("Invalid file type. Only DOCX files are supported.")
                
                self.log(f"Processing DOCX file: {resolved_path.name}")
                text = self.extract_text_from_docx(resolved_path)
                chunks = self._create_semantic_chunks(text)
                keywords_list = self._process_keywords(text, resolved_path)
                
                for chunk_id, (chunk_type, content) in enumerate(chunks):
                    metadata = self._format_metadata(
                        f"file_chunk_{chunk_id}",
                        chunk_type,
                        resolved_path.name,
                        keywords_list
                    )
                    
                    formatted_content = (
                        f"Type: {chunk_type}\n"
                        f"Content:\n{content}\n"
                        f"Metadata: {metadata}"
                    )
                    data_chunks.append(Data(text=formatted_content))

            # Handle Data objects if provided
            if hasattr(self, "data") and self.data:
                self.log("Processing provided Data objects...")
                for data_idx, data_item in enumerate(self.data):
                    text = data_item.text
                    chunks = self._create_semantic_chunks(text)
                    keywords_list = self._process_keywords(text)
                    
                    for chunk_id, (chunk_type, content) in enumerate(chunks):
                        metadata = self._format_metadata(
                            f"data_{data_idx}_chunk_{chunk_id}",
                            chunk_type,
                            "data input",
                            keywords_list
                        )
                        
                        formatted_content = (
                            f"Type: {chunk_type}\n"
                            f"Content:\n{content}\n"
                            f"Metadata: {metadata}"
                        )
                        data_chunks.append(Data(text=formatted_content))

            if not data_chunks:
                self.log("No valid input provided for processing.")
                raise ValueError("Please provide a DOCX file path or a list of Data objects.")

            self.log(f"Successfully processed inputs into {len(data_chunks)} semantic chunks.")
            return data_chunks

        except Exception as e:
            self.log(f"Error processing input: {str(e)}")
            if not self.silent_errors:
                raise
            return []