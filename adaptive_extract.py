"""OCR tối đa 5 vùng chữ, dùng toàn trang để bổ sung trường còn thiếu."""
import sys


# Trường bắt buộc theo từng loại giấy; đường dẫn là các khóa lồng nhau.
REQUIRED = {
    'ma_so_doanh_nghiep': (('ma_so_doanh_nghiep',), ('ten_cong_ty', 'tieng_viet'),
                           ('tru_so_chinh', 'dia_chi'), ('von_dieu_le',)),
    'ma_so_ho_kinh_doanh': (('ten_ho_kinh_doanh',), ('dia_diem_kinh_doanh', 'dia_chi'),
                            ('chu_ho', 'ho_ten')),
}


def _empty(value):
    if isinstance(value, str):
        return not value.strip()
    return value is None or value == [] or value == {}


def _get(data, path):
    for key in path:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def missing_fields(data):
    kind = next((key for key in REQUIRED if key in data), None)
    missing = ['.'.join(path) for path in REQUIRED.get(kind, ()) if _empty(_get(data, path))]
    people = data.get('thanh_vien', []) + data.get('nguoi_dai_dien', [])
    if 'nguoi_dai_dien' in data and not any(not _empty(p.get('ho_ten')) for p in people):
        missing.append('thanh_vien/nguoi_dai_dien')
    return missing


def _identity(person):
    """Số giấy tờ ổn định hơn tên có dấu bị OCR sai."""
    if not isinstance(person, dict):
        return None
    number = person.get('so_giay_to')
    if number:
        return 'id', str(number).strip()
    name = person.get('ho_ten')
    return ('name', ' '.join(name.casefold().split())) if name else None


def fill_missing(existing, recovered):
    """Điền dữ liệu trống, giữ nguyên giá trị đã đọc được từ các vùng."""
    if _empty(existing):
        return recovered
    if isinstance(existing, dict) and isinstance(recovered, dict):
        return {key: fill_missing(value, recovered.get(key)) for key, value in existing.items()}
    if isinstance(existing, list) and isinstance(recovered, list):
        # Ghép người theo số giấy tờ hoặc tên duy nhất, không theo vị trí.
        keys = [_identity(item) for item in existing]
        merged = []
        for item, key in zip(existing, keys):
            matches = [c for c in recovered if key and _identity(c) == key]
            merged.append(fill_missing(item, matches[0])
                          if keys.count(key) == 1 and len(matches) == 1 else item)
        return merged
    return existing


def adaptive_extract(read_pages, extract, read_full_pages):
    """extract(text) -> dict; lỗi ValueError coi như chưa trích xuất được."""
    pages = read_pages(5)
    text = '\n\n'.join(page['text'] for page in pages) + '\n'
    try:
        data = extract(text)
    except ValueError:
        data = None
        missing = ['chưa nhận diện/trích xuất được tài liệu']
    else:
        missing = missing_fields(data)
        if not missing:
            return pages, data
    print(f"Còn thiếu {', '.join(missing)}; OCR toàn trang", file=sys.stderr)
    pages = read_full_pages()
    text = '\n\n'.join(page['text'] for page in pages) + '\n'
    recovered = extract(text)
    data = recovered if data is None else fill_missing(data, recovered)
    remaining = missing_fields(data)
    if remaining:
        print('Sau OCR toàn trang vẫn còn trống: ' + ', '.join(remaining), file=sys.stderr)
    return pages, data
