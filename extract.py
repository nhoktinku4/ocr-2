"""Trích xuất văn bản theo mẫu JSON đơn giản: object, array và chuỗi."""
import re
import unicodedata


def template_schema(template):
    if isinstance(template, dict):
        return {'type': 'object', 'properties': {k: template_schema(v) for k, v in template.items()},
                'required': list(template), 'additionalProperties': False}
    if isinstance(template, list) and len(template) == 1:
        return {'type': 'array', 'items': template_schema(template[0])}
    if isinstance(template, str):
        return {'type': 'string'}
    raise ValueError('Mẫu JSON chỉ chứa object, chuỗi và array có một phần tử mẫu.')


def validate_shape(value, template, path='$'):
    if isinstance(template, dict):
        if not isinstance(value, dict) or set(value) != set(template):
            raise ValueError(f'{path}: các trường không khớp mẫu JSON')
        for key in template:
            validate_shape(value[key], template[key], f'{path}.{key}')
    elif isinstance(template, list):
        if not isinstance(value, list):
            raise ValueError(f'{path}: phải là array')
        for index, item in enumerate(value):
            validate_shape(item, template[0], f'{path}[{index}]')
    elif not isinstance(value, str):
        raise ValueError(f'{path}: phải là chuỗi')


def without_accents(value):
    """Chuẩn hóa để nhận diện nhãn, không dùng để sửa giá trị OCR."""
    value = value.replace('đ', 'd').replace('Đ', 'D')
    return ''.join(c for c in unicodedata.normalize('NFD', value)
                   if not unicodedata.combining(c))


_VIETNAMESE = 'aàáảãạăằắẳẵặâầấẩẫậeèéẻẽẹêềếểễệiìíỉĩịoòóỏõọôồốổỗộơờớởỡợuùúủũụưừứửữựyỳýỷỹỵdđ'
_VARIANTS = {}
for _char in _VIETNAMESE:
    _VARIANTS.setdefault(without_accents(_char), []).append(_char)


def label_regex(pattern, flags=re.I):
    """Mở rộng chữ trong regex thành nhóm có/không dấu; giữ nguyên escape.

    Match trực tiếp trên văn bản gốc nên vị trí cắt và giá trị có dấu không đổi.
    """
    pieces = []
    index = 0
    while index < len(pattern):
        char = pattern[index]
        if char == '\\' and index + 1 < len(pattern):
            pieces.append(pattern[index:index + 2])
            index += 2
            continue
        base = without_accents(char).lower()
        if base in _VARIANTS:
            pieces.append('[' + ''.join(_VARIANTS[base]) + ']')
        else:
            pieces.append(char)
        index += 1
    return re.compile(''.join(pieces), flags)


# Nhãn là ranh giới để không lấy nhầm giá trị của trường kế tiếp.
LABELS = {
    'ma_so_doanh_nghiep': r'Mã số doanh nghiệp',
    'so_giay_chung_nhan': r'Số',
    'ten_ho_kinh_doanh': r'Tên hộ kinh doanh(?:\s*\(ghi bằng chữ in hoa\))?',
    'tieng_viet': r'Tên công ty viết bằng tiếng Việt',
    'tieng_nuoc_ngoai': r'Tên công ty viết bằng tiếng nước ngoài',
    'viet_tat': r'Tên công ty viết tắt',
    'dia_chi': r'(?:Địa chỉ trụ sở(?: hộ kinh doanh| chính)?|Địa điểm(?: trụ sở hộ)? kinh doanh)',
    'dien_thoai': r'Điện thoại', 'fax': r'Fax', 'email': r'Email', 'website': r'Website',
    'nganh_nghe': r'Ngành\s*,?\s*nghề kinh doanh',
    'von': r'Vốn (?:kinh doanh|điều lệ)(?:\s*\(Bằng số\))?',
    'bang_chu': r'(?:Vốn (?:kinh doanh|điều lệ)\s*\(Bằng chữ\)|Bằng chữ)', 'menh_gia_co_phan': r'Mệnh giá cổ phần',
    'tong_so_co_phan': r'Tổng số cổ phần',
    'chu_the_thanh_lap': r'Chủ thể thành lập hộ kinh doanh',
    'ho_ten': r'Họ và tên(?: đại diện hộ kinh doanh)?', 'gioi_tinh': r'Giới tính', 'chuc_danh': r'Chức danh',
    'ngay_sinh': r'(?:Sinh ngày|Ngày sinh)', 'dan_toc': r'Dân tộc', 'quoc_tich': r'Quốc tịch',
    'loai_giay_to': r'Loại giấy tờ (?:pháp lý(?: của cá nhân)?|chứng thực cá nhân)',
    'so_giay_to': r'(?:Số giấy tờ (?:pháp lý(?: của cá nhân)?|chứng thực cá nhân)|Chứng minh nhân dân số|Căn cước công dân số|Số giấy chứng thực cá nhân|Số căn cước công dân(?:\s*/\s*chứng minh nhân dân)?|CCCD/CMND|Số CCCD|Số CMND)',
    'ngay_cap': r'Ngày cấp', 'noi_cap': r'(?:Nơi cấp|Cơ quan cấp)',
    'dia_chi_thuong_tru': r'(?:Địa chỉ thường trú|Nơi đăng ký hộ khẩu thường trú)',
    'dia_chi_lien_lac': r'(?:Địa chỉ liên lạc|Chỗ ở hiện tại)',
}
BOUNDARY = label_regex(
    r'(?:' + '|'.join(LABELS.values()) + r')\s*:'
    r'|^[ \t]*(?:#{1,6}[ \t]*)?(?:\d+\.[ \t]*)?' + LABELS['dia_chi'] + r'[ \t]*\r?$'
    r'|^\s*\d+\.\s+'
    r'|^\s*(?:\d+\.){8,}'
    r'|^\s*(?:KT\.\s*)?(?:PHÓ\s+)?TRƯỞNG PHÒNG'
    r'|\bKT\s*\.\s*(?:PHÓ\s+)?(?:TRƯỞNG|TƯỞNG)\s+PHÒNG\b'
    r'|^[ \t]*(?:Kí|Ký)\s*:\s*(?:KT\.\s*)?(?:PHÓ\s+)?TRƯỞNG PHÒNG'
    r'|^\s*Đăng ký (?:lần|thay đổi)'
    r'|^\s*Giấy tờ chứng thực cá nhân khác[^\n]*:'
    r'|\bChứng thực\s+bản sao\s+đúng\s+với\s+bản chính\b'
    r'|\bSố\s+chứng thực\s*:'
    r'|\bGhi\s+chú\s*:'
    r'|\bĐã\s+đối\s+chiếu\s+bản\s+chính\b'
    r'|^--- Trang \d+ ---', re.I | re.M)


def clean(value):
    value = re.sub(r'<br\s*/?>', ' ', value, flags=re.I)
    return re.sub(r'\s+', ' ', value).strip(' \t\r\n.:;')


def cut_address(value):
    """Giữ địa chỉ đến hết tên quốc gia, bỏ nội dung OCR phía sau."""
    country = label_regex(r'\bViệt\s+Nam\b').search(value)
    return value[:country.end()] if country else value


def field(text, key, raw=False):
    match = label_regex(LABELS[key] + r'\s*:').search(text)
    if match is None and key in ('dia_chi', 'von'):
        # Địa chỉ và vốn có thể là đề mục riêng, không có dấu hai chấm.
        match = label_regex(
            r'^\s*(?:#{1,6}\s*)?(?:\d+\.\s*)?' + LABELS[key]
            + r'[ \t]*\r?\n', re.I | re.M
        ).search(text)
    if not match:
        return ''
    end = BOUNDARY.search(text, match.end())
    value = text[match.end():end.start() if end else len(text)]
    # Dòng trống ngăn cách các vùng OCR độc lập.
    value = re.split(r'\r?\n[ \t]*\r?\n', value, maxsplit=1)[0]
    if key in ('dia_chi', 'dia_chi_thuong_tru', 'dia_chi_lien_lac'):
        value = cut_address(value)
        watermark = re.search(r'\s*(?:\d+(?:[.,]\d+)?\s*%\s*)?\bCamScanner\b', value, re.I)
        if watermark:
            value = value[:watermark.start()]
        # Nhãn điện thoại bị OCR sai trong một số bản quét.
        phone_noise = label_regex(r'^[ \t]*Diễn theo\s*:\s*\d[\d \t.-]*$', re.I | re.M).search(value)
        if phone_noise:
            value = value[:phone_noise.start()]
    if key == 'ma_so_doanh_nghiep':
        code = re.match(r'\s*([0-9]+)\b', value)
        return code[1] if code else ''
    return value if raw else clean(value)


def project(template, values):
    """Chỉ xuất khóa trong mẫu; khóa chưa có quy tắc được để trống."""
    if isinstance(template, dict):
        values = values if isinstance(values, dict) else {}
        return {key: project(value, values.get(key)) for key, value in template.items()}
    if isinstance(template, list):
        return [project(template[0], item) for item in (values or [])]
    return values if isinstance(values, str) else ''


def table_members(text):
    """Đọc bảng thành viên Markdown, giữ cột cho bảng tiếp nối qua trang."""
    members, columns = [], None
    for line in text.splitlines():
        stripped = line.strip()
        if not stripped.startswith('|'):
            if re.match(r'^\s*\d+\.\s+', line):
                columns = None
            continue
        cells = [clean(cell) for cell in stripped.strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', cell.replace(' ', '')) for cell in cells):
            continue
        normalized = [without_accents(cell).lower() for cell in cells]
        if any('ten thanh vien' in cell for cell in normalized):
            columns = {}
            for index, cell in enumerate(normalized):
                if 'ten thanh vien' in cell:
                    key = 'ho_ten'
                elif 'quoc tich' in cell:
                    key = 'quoc_tich'
                elif 'dia chi' in cell:
                    key = 'dia_chi_thuong_tru'
                elif 'von gop' in cell:
                    key = 'phan_gop_von'
                elif 'ty le' in cell:
                    key = 'ty_le'
                elif ('giay to' in cell or 'ma so doanh nghiep' in cell
                      or 'so cmnd' in cell or 'so cccd' in cell):
                    key = 'so_giay_to'
                elif 'ghi chu' in cell:
                    key = 'ghi_chu'
                else:
                    continue
                columns[key] = index
            continue
        # Không coi header bảng khác hoặc dòng phân cách là thành viên.
        if columns is None or not cells or not cells[0].isdigit():
            continue
        person = {key: cells[index] if index < len(cells) else ''
                  for key, index in columns.items()}
        if 'dia_chi_thuong_tru' in person:
            person['dia_chi_thuong_tru'] = clean(cut_address(person['dia_chi_thuong_tru']))
        if person.get('ho_ten'):
            members.append(person)
    return members


def table_industries(text):
    """Đọc bảng ngành nghề Markdown theo tên cột, giữ nguyên nội dung OCR."""
    industries, columns = [], None
    for line in text.splitlines():
        if not line.strip().startswith('|'):
            continue
        cells = [clean(cell) for cell in line.strip().strip('|').split('|')]
        if all(re.fullmatch(r':?-+:?', cell.replace(' ', '')) for cell in cells):
            continue
        normalized = [without_accents(cell).lower() for cell in cells]
        if 'ten nganh' in normalized and 'ma nganh' in normalized:
            columns = (normalized.index('ten nganh'), normalized.index('ma nganh'))
            continue
        if columns is None or max(columns) >= len(cells):
            continue
        name, code = (cells[index] for index in columns)
        match = label_regex(r'^(\d{4,5})(?:\s*\(Chính\))?$').fullmatch(code)
        if name and (match or not code):
            industries.append({'ten_nganh_nghe': name, 'ma_nganh': match[1] if match else ''})
    return industries


def extract(text, template):
    template_schema(template)
    text = unicodedata.normalize('NFC', text)
    if not text.strip():
        raise ValueError('Văn bản OCR rỗng; không thể trích xuất.')
    values = {key: field(text, key) for key in LABELS}
    # Ngày được giữ theo cách ghi trên tài liệu; không suy đoán ngày/tháng còn thiếu.
    first = label_regex(r'Đăng ký (?:lần )?đầu[^\n]*').search(text)
    values['ngay_dang_ky_lan_dau'] = ''
    if first:
        date = label_regex(r'ngày\s+(.+)').search(first.group())
        values['ngay_dang_ky_lan_dau'] = clean(date[1]) if date else ''
    change = label_regex(r'Đăng ký thay đổi lần thứ\s*:\s*(\d+)\s*,?\s*(?:ngày\s*)?([^\n]+)').search(text)
    values['lan_thay_doi'] = change[1] if change else ''
    values['ngay_dang_ky_thay_doi'] = clean(change[2]) if change else ''
    values['ten_cong_ty'] = {k: values[k] for k in ('tieng_viet', 'tieng_nuoc_ngoai', 'viet_tat')}
    contact = {k: values[k] for k in ('dia_chi', 'dien_thoai', 'fax', 'email', 'website')}
    values['tru_so_chinh'] = contact
    values['tru_so_ho_kinh_doanh'] = contact
    capital = values['von']
    if not capital:
        capital = field(text, 'von', raw=True)
        numbered = label_regex(LABELS['von'] + r'\s*:').finditer(text)
        capital = next((field(text[m.start():], 'von') for m in numbered
                        if label_regex(r'\(Bằng số\)').search(m.group())), capital)
    amount = re.match(r'([\d][\d., \t]*(?:đồng\b)?)', capital, re.I)
    words = re.search(r'\(([^()]+)\)', capital)
    money = {'so_tien': clean(amount[1]) if amount else '',
             'bang_chu': values['bang_chu'] or (clean(words[1]) if words else ''),
             'menh_gia_co_phan': label_regex(r'\s*đồng\.?$').sub('', values['menh_gia_co_phan']),
             'tong_so_co_phan': values['tong_so_co_phan']}
    values['von_dieu_le'] = money
    values['von_kinh_doanh'] = money
    # Vai trò theo đề mục, giữ xuyên trang; không suy ra từ tên hay chức danh.
    sections = list(label_regex(
        r'^\s*(?:#{1,6}\s*)?\d+\.\s+([^\n]+)', re.I | re.M
    ).finditer(text))
    people, members, household = [], [], []
    matches = list(label_regex(LABELS['ho_ten'] + r'\s*:').finditer(text))
    for index, match in enumerate(matches):
        end = matches[index + 1].start() if index + 1 < len(matches) else len(text)
        next_section = next((s for s in sections if s.start() > match.start()), None)
        if next_section:
            end = min(end, next_section.start())
        block = text[match.start():end]
        person = {k: field(block, k) for k in ('ho_ten', 'gioi_tinh', 'chuc_danh', 'ngay_sinh',
            'dan_toc', 'quoc_tich', 'loai_giay_to', 'so_giay_to', 'ngay_cap', 'noi_cap',
            'dia_chi_thuong_tru', 'dia_chi_lien_lac')}
        person['CCCD/CMND'] = person['so_giay_to']
        heading = next((s[1] for s in reversed(sections) if s.start() < match.start()), '')
        heading = without_accents(heading).lower()
        if 'chu so huu' in heading or 'thanh vien' in heading:
            members.append(person)
        elif 'nguoi dai dien theo phap luat' in heading:
            people.append(person)
        elif re.search(r'chu ho(?: ho)? kinh doanh', heading) or 'dai dien ho kinh doanh' in heading:
            household.append(person)
        elif not sections and 'chu_ho_kinh_doanh' in template:
            household.append(person)
    values['thanh_vien'] = members + table_members(text)
    values['nguoi_dai_dien_theo_phap_luat'] = people
    owner = dict(household[0]) if household else {}
    # Mục chủ hộ chỉ có một người; OCR có thể lặp nhãn họ tên giữa mục.
    for continuation in household[1:]:
        for key, value in continuation.items():
            if not owner.get(key):
                owner[key] = value
    values['chu_ho_kinh_doanh'] = owner
    # Bảng OCR có thể đứng riêng hoặc cách nhãn ngành nghề bởi dòng trống.
    industries = table_industries(text)
    raw = label_regex(r'\(Hộ kinh doanh').split(field(text, 'nganh_nghe', raw=True), maxsplit=1)[0]
    markdown_header = label_regex(
        r'^\s*\|[^\n]*Tên ngành[^\n]*\|', re.I | re.M).search(raw)
    table_header = label_regex(r'^\s*STT\s+Tên ngành\s+Mã ngành\s*$', re.I | re.M).search(raw)
    if industries or markdown_header:
        raw = ''
    elif table_header:
        rows = raw[table_header.end():]
        for row in re.split(r'^[ \t]*\d+[ \t]+(?=[^\d\s(])', rows, flags=re.M):
            match = re.fullmatch(r'\s*(.+?)\s+(\d{4,5})(?:\s*\(Chính\))?\s*', row, re.S | re.I)
            if match:
                industries.append({'ten_nganh_nghe': clean(match[1]), 'ma_nganh': match[2]})
        raw = ''
    else:
        raw = clean(raw)
    for line in re.split(r'[\n;]+', raw):
        match = re.search(r'(.+?)\s*\(([\d,\s]+)\)', line)
        if match:
            industries.append({'ten_nganh_nghe': clean(match[1]), 'ma_nganh': clean(match[2])})
        elif clean(line):
            industries.append({'ten_nganh_nghe': clean(line), 'ma_nganh': ''})
    values['nganh_nghe_kinh_doanh'] = industries
    data = project(template, values)
    validate_shape(data, template)
    return data
 