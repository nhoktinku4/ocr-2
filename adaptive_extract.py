"""OCR tối đa 5 vùng chữ, dùng toàn trang để bổ sung trường còn thiếu."""
import sys

from extract import extract


def missing_fields(data):
    missing = []
    for key in ('ma_so_doanh_nghiep', 'ten_ho_kinh_doanh'):
        if key in data and not data[key].strip():
            missing.append(key)
    for key, child in (('ten_cong_ty', 'tieng_viet'),
                       ('tru_so_chinh', 'dia_chi'), ('von_dieu_le', 'so_tien'),
                       ('von_kinh_doanh', 'so_tien'),
                       ('tru_so_ho_kinh_doanh', 'dia_chi')):
        if key in data and not data[key].get(child, '').strip():
            missing.append(key)
    owner = data.get('chu_ho_kinh_doanh')
    if owner is not None and any(not value.strip() for value in (owner.get('ho_ten', ''), *owner.values())):
        missing.append('chu_ho_kinh_doanh')
    for key, child in (('nganh_nghe_kinh_doanh', 'ten_nganh_nghe'),):
        if key in data and not any(item.get(child, '').strip() for item in data[key]):
            missing.append(key)
    people_keys = ('thanh_vien', 'nguoi_dai_dien_theo_phap_luat')
    if any(key in data for key in people_keys) and not any(
        person.get('ho_ten', '').strip()
        for key in people_keys for person in data.get(key, [])
    ):
        missing.extend(key for key in people_keys if key in data)
    else:
        for key in people_keys:
            # Bảng thành viên chỉ cần có tên; các thông tin khác có thể không có trên giấy.
            if key == 'thanh_vien':
                continue
            if any(any(not value.strip() for field, value in person.items()
                       if field != 'ho_ten')
                   for person in data.get(key, [])):
                missing.append(key)
    return missing


def fill_missing(existing, recovered):
    """Điền dữ liệu trống, giữ nguyên giá trị đã đọc được từ các vùng."""
    if type(existing) is not type(recovered):
        return existing
    if isinstance(existing, dict):
        return {key: fill_missing(value, recovered.get(key, value))
                for key, value in existing.items()}
    if isinstance(existing, list):
        # Ghép hàng theo tên duy nhất, không theo vị trí có thể thay đổi sau OCR.
        for name_key in ('ten_nganh_nghe', 'ho_ten'):
            if existing and all(isinstance(item, dict) and name_key in item
                                for item in existing):
                if not any(item[name_key].strip() for item in existing):
                    return recovered
                merged = []
                for item in existing:
                    # Số giấy tờ ổn định hơn tên có dấu bị OCR sai.
                    identity = item.get('so_giay_to', '').strip()
                    id_matches = [candidate for candidate in recovered
                                  if isinstance(candidate, dict) and identity
                                  and candidate.get('so_giay_to', '').strip() == identity]
                    if (identity and len(id_matches) == 1
                            and sum(row.get('so_giay_to', '').strip() == identity
                                    for row in existing) == 1):
                        merged.append(fill_missing(item, id_matches[0]))
                        continue
                    name = ' '.join(item[name_key].casefold().split())
                    matches = [candidate for candidate in recovered
                               if isinstance(candidate, dict) and name
                               and ' '.join(candidate.get(name_key, '').casefold().split()) == name]
                    unique = sum(' '.join(row[name_key].casefold().split()) == name
                                 for row in existing) == 1
                    merged.append(fill_missing(item, matches[0])
                                  if unique and len(matches) == 1 else item)
                return merged
        if not existing or not any(_has_value(item) for item in existing):
            return recovered
        return existing
    return existing if existing.strip() else recovered


def _has_value(value):
    if isinstance(value, dict):
        return any(_has_value(item) for item in value.values())
    if isinstance(value, list):
        return any(_has_value(item) for item in value)
    return bool(value.strip())


def adaptive_extract(read_pages, select_template, read_full_pages):
    pages = read_pages(5)
    text = '\n\n'.join(page['text'] for page in pages) + '\n'
    try:
        template = select_template(text)
        data = extract(text, template)
    except ValueError:
        template = None
        missing = ['chưa nhận diện/trích xuất được tài liệu']
    else:
        missing = missing_fields(data)
        if not missing:
            return pages, data
    print(f"Còn thiếu {', '.join(missing)}; OCR toàn trang", file=sys.stderr)
    pages = read_full_pages()
    text = '\n\n'.join(page['text'] for page in pages) + '\n'
    if template is None:
        data = extract(text, select_template(text))
    else:
        recovered = extract(text, template)
        data = fill_missing(data, recovered)
    remaining = missing_fields(data)
    if remaining:
        print('Sau OCR toàn trang vẫn còn trống: ' + ', '.join(remaining), file=sys.stderr)
    return pages, data
 