"""Nhận diện loại đăng ký từ tiêu đề và nhãn trong văn bản OCR."""

import re

from extract import without_accents


def classify_document(text):
    normalized = re.sub(r'\s+', ' ', without_accents(text).lower())
    # Ưu tiên tiêu đề giấy chứng nhận, tránh nhầm các nội dung được nhắc tới.
    titles = {
        'doanh_nghiep': r'giay chung nhan dang ky doanh nghiep',
        'ho_kinh_doanh': r'giay chung nhan dang ky ho kinh doanh',
    }
    matches = [kind for kind, pattern in titles.items() if re.search(pattern, normalized)]
    if len(matches) == 1:
        return matches[0]
    if not matches:
        markers = {
            'doanh_nghiep': ('ma so doanh nghiep', 'ten cong ty viet bang tieng viet'),
            'ho_kinh_doanh': ('ten ho kinh doanh', 'chu the thanh lap ho kinh doanh'),
        }
        matches = [kind for kind, labels in markers.items()
                   if any(label in normalized for label in labels)]
        if len(matches) == 1:
            return matches[0]
    raise ValueError('Không xác định được loại tài liệu hoặc có dấu hiệu cả hai loại; '
                     'hãy chọn mẫu bằng --template.')
 