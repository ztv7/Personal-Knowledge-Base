from typing import List, Optional, Dict, Any, Tuple
from src.config import settings

VALID_EXTENSIONS = {".pdf", ".txt", ".md"}
MAX_FILE_SIZE_BYTES = settings.max_file_size_mb * 1024 * 1024


class FileValidationError(Exception):
    pass


def validate_file(filename: str, content_bytes: bytes) -> Tuple[str, str]:
    """
    验证文件合法性，返回 (extension, safe_filename)
    raises FileValidationError on invalid input
    """
    if not filename or "." not in filename:
        raise FileValidationError("文件名不合法，缺少扩展名")

    ext = filename[filename.rfind("."):].lower()
    if ext not in VALID_EXTENSIONS:
        raise FileValidationError(
            f"不支持的文件格式 '{ext}'，仅支持 PDF/TXT/MD"
        )

    if len(content_bytes) == 0:
        raise FileValidationError("文件内容为空")

    if len(content_bytes) > MAX_FILE_SIZE_BYTES:
        raise FileValidationError(
            f"文件过大({len(content_bytes)/1024/1024:.1f}MB)，"
            f"最大允许 {settings.max_file_size_mb}MB"
        )

    return ext, filename
