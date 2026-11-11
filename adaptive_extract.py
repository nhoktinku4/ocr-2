"""OCR tối đa 5 vùng chữ, dùng toàn trang để bổ sung trường còn thiếu."""
from logger import logger


# Trường bắt buộc theo từng loại giấy; đường dẫn là các khóa lồng nhau.
REQUIRED = {
    'ma_so_doanh_nghiep': (('ma_so_doanh_nghiep',), ('dang_ky_lan_dau',),
                           ('ten_cong_ty', 'tieng_viet'), ('tru_so_chinh', 'dia_chi'),
                           ('von_dieu_le',), ('co_quan_cap',)),
    'ma_so_ho_kinh_doanh': (('ten_ho_kinh_doanh',), ('dang_ky_lan_dau',),
                            ('dia_diem_kinh_doanh', 'dia_chi'), ('von_kinh_doanh',),
                            ('chu_ho', 'ho_ten'), ('co_quan_cap',)),
}
# Thành viên: có 2 mẫu, mỗi mẫu được phép trống các trường riêng, còn lại phải đủ.
# A: chủ sở hữu ghi dạng đoạn văn; B: dòng bảng góp vốn (có phan_von_gop hoặc ty_le).
MEMBER_OPTIONAL_A = {'phan_von_gop', 'ty_le', 'ghi_chu'}
MEMBER_OPTIONAL_B = {'gioi_tinh', 'ngay_sinh', 'dan_toc', 'quoc_tich', 'ghi_chu', 'ngay_cap',
                     'noi_cap', 'dia_chi_lien_lac'}


def _empty(value):
    if isinstance(value, str):
        return not value.strip()
    if isinstance(value, dict):                 # vd. co_quan_cap: mọi trường con đều trống
        return all(_empty(v) for v in value.values())
    return value is None or value == []


def _get(data, path):
    for key in path:
        data = data.get(key) if isinstance(data, dict) else None
    return data


def missing_fields(data):
    kind = next((key for key in REQUIRED if key in data), None)
    missing = ['.'.join(path) for path in REQUIRED.get(kind, ()) if _empty(_get(data, path))]
    # Người đại diện / chủ hộ: đọc được một phần thì đọc lại cho đủ; trống hết thì thôi
    # (chủ hộ trống hết đã bị bắt qua REQUIRED chu_ho.ho_ten).
    people = [(f'nguoi_dai_dien[{i}]', p) for i, p in enumerate(data.get('nguoi_dai_dien') or [])]
    people.append(('chu_ho', data.get('chu_ho')))
    for name, person in people:
        if isinstance(person, dict) and not _empty(person):
            missing += [f'{name}.{key}' for key, value in person.items()
                        if _empty(value) and f'{name}.{key}' not in missing]
    for i, person in enumerate(data.get('thanh_vien') or []):
        if isinstance(person, dict) and not _empty(person):
            empty = {key for key, value in person.items() if _empty(value)}
            # có phần vốn góp / tỷ lệ -> dòng bảng (B), không có -> đoạn văn (A)
            in_table = not (_empty(person.get('phan_von_gop')) and _empty(person.get('ty_le')))
            lacking = empty - (MEMBER_OPTIONAL_B if in_table else MEMBER_OPTIONAL_A)
            missing += [f'thanh_vien[{i}].{key}' for key in person if key in lacking]
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
    if recovered is None:
        return existing
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
    pages = read_pages(9)
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
    logger.info(f"Còn thiếu {', '.join(missing)}; OCR toàn trang")
    pages = read_full_pages()
    text = '\n\n'.join(page['text'] for page in pages) + '\n'
    recovered = extract(text)
    # Hai lần đọc nhận ra loại giấy khác nhau thì tin toàn trang (đủ ngữ cảnh hơn).
    if data is None or data.get('loai_giay_to') != recovered.get('loai_giay_to'):
        data = recovered
    else:
        data = fill_missing(data, recovered)
    remaining = missing_fields(data)
    if remaining:
        logger.warning('Sau OCR toàn trang vẫn còn trống: ' + ', '.join(remaining))
    return pages, data
