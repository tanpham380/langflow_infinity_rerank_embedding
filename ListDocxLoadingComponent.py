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

    def process_docx(self) -> List[Data]:
        data_chunks = []

        try:
            # Xử lý file DOCX từ file path (nếu được cung cấp)
            if self.path:
                resolved_path = Path(self.resolve_path(self.path))
                if resolved_path.suffix.lower() != ".docx":
                    raise ValueError("Invalid file type. Only DOCX files are supported.")
                self.log(f"Processing DOCX file: {resolved_path.name}")
                text = self._extract_text_from_docx(resolved_path)
                chunks = self._create_semantic_chunks(text)
                keywords_list = self._process_keywords(text, resolved_path)
                keywords_tag = " ".join([f"#{kw}" for kw in keywords_list])
                for chunk_id, (chunk_type, content) in enumerate(chunks):
                    metadata = {
                        "chunk_id": f"file_chunk_{chunk_id}",
                        "chunk_type": chunk_type,
                        "source": resolved_path.name,
                        "keywords": keywords_tag
                    }
                    formatted_content = (
                        f"Type: {chunk_type}\n"
                        f"Content:\n{content}\n"
                        f"Metadata: {metadata}"
                    )
                    data_chunks.append(Data(text=formatted_content))
            
            # Xử lý danh sách Data objects (nếu được cung cấp)
            if hasattr(self, "data") and self.data:
                self.log("Processing provided Data objects...")
                for data_idx, data_item in enumerate(self.data):
                    text = data_item.text
                    chunks = self._create_semantic_chunks(text)
                    keywords_list = self._process_keywords(text)  # Không có file_path
                    keywords_tag = " ".join([f"#{kw}" for kw in keywords_list])
                    for chunk_id, (chunk_type, content) in enumerate(chunks):
                        metadata = {
                            "chunk_id": f"data_{data_idx}_chunk_{chunk_id}",
                            "chunk_type": chunk_type,
                            "source": "data input",
                            "keywords": keywords_tag
                        }
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

    def _extract_text_from_docx(self, file_path: Path) -> str:
        try:
            doc = Document(file_path)
            paragraphs = [para.text.strip() for para in doc.paragraphs if para.text.strip()]
            return "\n".join(paragraphs)
        except Exception as e:
            self.log(f"Error extracting text from DOCX: {e}")
            if not self.silent_errors:
                raise
            return ""

    def _create_semantic_chunks(self, text: str) -> List[Tuple[str, str]]:
        """
        Tạo các chunk mang ý nghĩa ngữ cảnh dựa theo cấu trúc đã định nghĩa.
        Nếu gặp tiêu đề cho cùng 1 section (ví dụ "Học phí") thì hợp nhất nội dung.
        Đối với "Nội dung khóa học" sẽ được xử lý riêng qua _split_program_content.
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
        Xử lý phần "Nội dung khóa học" thành các chunk kiểu "program" theo cấu trúc:
        
         - Chunk 1: Header chung (bao gồm nội dung giới thiệu, trước khi gặp "Chương trình học")
           và danh sách các tên chương trình học, ví dụ:
             +NGHIỆP VỤ QUẢN LÝ NHÀ HÀNG KHÁCH SẠN 
             +NGHIỆP VỤ QUẢN LÝ NHÀ HÀNG
             +NGHIỆP VỤ QUẢN LÝ KHÁCH SẠN
           và dòng "Từ khoá:" (nếu có) được đưa xuống dưới danh sách tên.
         
         - Các chunk tiếp theo: Mỗi khối chi tiết của từng chương trình, bắt đầu từ dòng "Chương trình học X:" 
           đến trước khối tiếp theo.
        
        Ví dụ, nếu file DOCX có nội dung:
            ... (header)
            Từ khoá: nội dung khoá học, học gì
            Chương trình học 1: NGHIỆP VỤ QUẢN LÝ NHÀ HÀNG KHÁCH SẠN 
            Nội dung: ...
            Từ khóa: ...
            Chương trình học 2: NGHIỆP VỤ QUẢN LÝ NHÀ HÀNG
            Nội dung: ...
            Từ khóa: ...
            Chương trình học 3: NGHIỆP VỤ QUẢN LÝ KHÁCH SẠN
            Nội dung: ...
            Từ khóa: ...
        
        Thì sẽ tạo ra 4 chunk:
          - Chunk header: phần header (trước dòng "Chương trình học 1:") kèm danh sách các chương trình học và dòng "Từ khoá:" (nếu có)
          - 3 chunk tiếp theo: ứng với từng chương trình học.
        """
        lines = content.splitlines()
        header_lines = []
        keyword_line = ""
        program_start_idx = None
    
        # Tìm chỉ số dòng đầu tiên chứa "Chương trình học \d+:"
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
    
            # Tìm các tên chương trình học từ phần chương trình
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
        Xử lý và kết hợp các từ khóa từ các nguồn khác nhau.
        Nếu file_path được cung cấp thì thêm tên file.
        """
        keywords = set()
        
        if self.meta_keywords:
            keywords.update(kw.strip().lower() for kw in self.meta_keywords.split(","))
        
        if file_path:
            keywords.add(file_path.stem.lower())
        
        course_name = self._extract_course_name(text)
        if course_name:
            keywords.update(kw.strip().lower() for kw in course_name.split(","))
            
        for line in text.splitlines():
            if "Từ khóa:" in line:
                section_keywords = line.split("Từ khóa:")[1].strip()
                keywords.update(kw.strip().lower() for kw in section_keywords.split(","))
        
        return sorted(list(keywords))

    def _extract_course_name(self, text: str) -> str:
        pattern = r"(?:•\s*)?Tên khóa học:\s*(.+)"
        match = re.search(pattern, text, re.IGNORECASE)
        if match:
            return match.group(1).strip()
        return ""
