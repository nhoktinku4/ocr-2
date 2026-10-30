"""
Trích xuất thông tin Giấy chứng nhận đăng ký doanh nghiệp từ text.txt (kết quả OCR)
theo mẫu gcn_dkdn_template.json, bằng luật (không cần model).

Sử dụng:
    python extract_info.py batch_out/2743901/text.txt          # in ra màn hình
    python extract_info.py batch_out/2743901/text.txt -o a.json
    python extract_info.py batch_out                           # mọi */text.txt -> extract.json cạnh nó

Chịu lỗi OCR:
    - nhãn so khớp không dấu + gần đúng ('Tân công ty viết tắt', 'Dịạ chỉ', 'Sinh ngay');
    - nhãn dính vào dòng trước ('Việt Nam Địa chỉ liên lạc: ...') được tách ra;
    - giá trị bị xuống dòng được nối lại;
    - sửa dấu / chính tả các từ thông dụng (địa danh, đơn vị hành chính, giấy tờ, chức danh,
      từ trong tên công ty, số đọc bằng chữ) - chỉ khi chắc chắn, tên riêng giữ nguyên.
"""
import argparse
import json
import os
import re
import unicodedata
from difflib import SequenceMatcher

HERE = os.path.dirname(os.path.abspath(__file__))
TEMPLATE = os.path.join(HERE, "json", "doanh_nghiep.json")       # doanh nghiệp
TEMPLATE_HKD = os.path.join(HERE, "json", "ho_kinh_doanh.json")  # hộ kinh doanh


# ----------------------------------------------------------------- chuẩn hóa chữ
def norm(s):
    """Bỏ dấu, chữ thường, gộp khoảng trắng: dùng để so khớp."""
    return norm_map(s)[0]


def norm_map(s):
    """-> (chuỗi chuẩn hóa, vị trí ký tự gốc của từng ký tự chuẩn hóa)."""
    out, pos = [], []
    for i, ch in enumerate(s):
        ch = "d" if ch in "đĐ" else ch
        for c in unicodedata.normalize("NFD", ch):
            if unicodedata.category(c) == "Mn":
                continue
            c = " " if c.isspace() else c.lower()
            if c == " " and (not out or out[-1] == " "):
                continue
            out.append(c)
            pos.append(i)
    while out and out[-1] == " ":
        out.pop(), pos.pop()
    return "".join(out), pos


def clean(s):
    """Bỏ ký tự rác OCR (chữ Hán, ký hiệu lạ), gộp khoảng trắng."""
    s = "".join(c for c in s if c.isascii() or unicodedata.name(c, "").startswith("LATIN")
                or c in "–“”‘’")
    return re.sub(r"\s+", " ", s).strip(" -–|")


# Từ/cụm từ viết đúng: chữ OCR có cùng dạng không dấu sẽ được thay bằng dạng này.
# Chỉ đưa vào từ gần như không thể nhầm nghĩa khi bỏ dấu.
ADMIN_WORDS = [
    "Việt Nam", "Thành phố", "Tỉnh", "Quận", "Huyện", "Thị xã", "Phường", "Thị trấn",
    "Khóm", "Ấp", "Khu phố", "Thôn", "Đường", "Hẻm", "Ngõ", "Ngách", "Lô", "Tầng", "Tòa nhà",
    "Khu công nghiệp", "Khu đô thị",
    "Hồ Chí Minh", "Hà Nội", "Đà Nẵng", "Hải Phòng", "Cần Thơ", "Bạc Liêu", "Cà Mau",
    "Sóc Trăng", "Kiên Giang", "An Giang", "Đồng Tháp", "Vĩnh Long", "Trà Vinh", "Bến Tre",
    "Tiền Giang", "Long An", "Hậu Giang", "Bình Dương", "Đồng Nai", "Bà Rịa", "Vũng Tàu",
    "Tây Ninh", "Bình Phước", "Lâm Đồng", "Đà Lạt", "Khánh Hòa", "Nha Trang", "Bình Thuận",
    "Ninh Thuận", "Đắk Lắk", "Gia Lai", "Kon Tum", "Quảng Nam", "Quảng Ngãi", "Bình Định",
    "Phú Yên", "Thừa Thiên Huế", "Huế", "Quảng Trị", "Quảng Bình", "Hà Tĩnh", "Nghệ An",
    "Thanh Hóa", "Ninh Bình", "Nam Định", "Thái Bình", "Hưng Yên", "Hải Dương", "Bắc Ninh",
    "Bắc Giang", "Quảng Ninh", "Vĩnh Phúc", "Phú Thọ", "Thái Nguyên", "Lạng Sơn",
    "Hóc Môn", "Củ Chi", "Thủ Đức", "Gò Vấp", "Tân Bình", "Bình Thạnh", "Phú Nhuận",
    "Ninh Kiều", "Cái Khế", "Cai Lậy", "Cái Bè", "Mỹ Tho", "Gò Công",
]
DOC_WORDS = [
    "Thẻ căn cước công dân", "Căn cước công dân", "Thẻ căn cước", "Căn cước",
    "Giấy chứng minh nhân dân", "Chứng minh nhân dân", "Hộ chiếu",
    "Cục Cảnh sát quản lý hành chính về trật tự xã hội", "Công an",
    "Giám đốc", "Tổng giám đốc", "Phó giám đốc", "Chủ tịch hội đồng thành viên",
    "Chủ tịch hội đồng quản trị", "Chủ tịch công ty", "Chủ doanh nghiệp",
    "Kinh", "Nam", "Nữ",
]
COMPANY_WORDS = [
    "Công ty", "Trách nhiệm hữu hạn", "Một thành viên", "Hai thành viên trở lên", "Cổ phần",
    "Doanh nghiệp tư nhân", "Xây dựng", "Thương mại", "Dịch vụ", "Xuất nhập khẩu",
    "Sản xuất", "Đầu tư", "Phát triển", "Kỹ thuật", "Công nghệ", "Vận tải", "Du lịch",
    "Tư vấn", "Thiết kế", "Bất động sản", "Nông nghiệp", "Thủy sản", "Môi trường",
    "Giải pháp", "Quốc tế", "Tập đoàn", "Phát", "Việt",
]
MONEY_WORDS = [
    "một", "hai", "ba", "bốn", "năm", "sáu", "bảy", "tám", "chín", "mười", "mươi",
    "trăm", "nghìn", "ngàn", "triệu", "tỷ", "đồng", "lẻ", "linh", "lăm",
]
AGENCIES = {"so ke hoach va dau tu": "Sở Kế hoạch và Đầu tư",
            "so tai chinh": "Sở Tài chính",
            "phong dang ky kinh doanh": "Phòng Đăng ký kinh doanh"}


def _vocab(words):
    """-> {số từ: {dạng không dấu: dạng đúng}}, ưu tiên cụm dài."""
    out = {}
    for w in words:
        out.setdefault(len(w.split()), {})[norm(w)] = w
    return sorted(out.items(), reverse=True)


VOCAB = {"admin": _vocab(ADMIN_WORDS + DOC_WORDS),
         "company": _vocab(COMPANY_WORDS + ADMIN_WORDS),
         "money": _vocab(MONEY_WORDS)}


def fix_words(s, kind="admin"):
    """Sửa dấu theo từ điển, giữ kiểu hoa/thường của bản OCR."""
    if not s:
        return s
    words, out, i = s.split(), [], 0
    while i < len(words):
        for n, table in VOCAB[kind]:
            chunk = words[i:i + n]
            core = [re.sub(r"^\W+|\W+$", "", w) for w in chunk]
            right = table.get(norm(" ".join(core))) if len(chunk) == n and all(core) else None
            if right and n == 1 and kind != "money":
                # từ đơn dễ nhầm ('Dương' / 'Đường', 'quản' / 'Quận'): chỉ sửa chữ OCR
                # mất hết dấu; đơn vị hành chính phải đứng trước tên riêng / số
                nxt = words[i + n] if i + n < len(words) else ""
                ok = core[0].isascii()
                if kind == "admin" and len(words) > 1:
                    ok = (ok and (nxt[:1].isupper() or nxt[:1].isdigit())) or nxt[:1].isdigit()
                right = right if ok else None
            if right:
                lead = re.match(r"^\W*", chunk[0])[0]
                tail = re.search(r"\W*$", chunk[-1])[0]
                src = " ".join(core)
                if src.isupper():
                    right = right.upper()
                elif src[:1].isupper() and not right[:1].isupper():
                    right = right[:1].upper() + right[1:]
                out.append(lead + right + tail)
                i += n
                break
        else:
            out.append(words[i])
            i += 1
    return " ".join(out)


# ----------------------------------------------------------------- nhãn
LABELS = {
    "ma_so": ["ma so doanh nghiep"],
    "lan_dau": ["dang ky lan dau"],
    "thay_doi": ["dang ky thay doi lan thu", "dang ky thay doi"],
    "ten_vn": ["ten cong ty viet bang tieng viet"],
    "ten_nn": ["ten cong ty viet bang tieng nuoc ngoai"],
    "ten_vt": ["ten cong ty viet tat"],
    "dien_thoai": ["dien thoai", "telephone", "phone", "tel", "so dien thoai"],
    "fax": ["fax"], "email": ["email", "e-mail"], "website": ["website", "web site"],
    "von": ["von dieu le"],
    "ma_so_hkd": ["ma so ho kinh doanh"],
    "so_dk_hkd": ["ma so dang ky ho kinh doanh", "so giay chung nhan dang ky ho kinh doanh"],
    "ten_hkd": ["ten ho kinh doanh", "ten ho kinh doanh viet bang tieng viet"],
    "dia_diem": ["dia diem kinh doanh", "dia chi tru so ho kinh doanh", "dia chi kinh doanh"],
    "von_kd": ["von kinh doanh"],
    "nganh": ["nganh nghe kinh doanh", "nganh, nghe kinh doanh"],
    "bang_chu": ["bang chu"],
    "ho_ten": ["ho va ten", "ho ten", "ho va ten dai dien ho kinh doanh"],
    "gioi_tinh": ["gioi tinh"],
    "chuc_danh": ["chuc danh"],
    "ngay_sinh": ["sinh ngay", "ngay sinh"],
    "dan_toc": ["dan toc"],
    "quoc_tich": ["quoc tich"],
    "loai_giay_to": ["loai giay to phap ly cua ca nhan", "loai giay to chung thuc ca nhan",
                     "loai giay to"],
    "so_giay_to": ["so giay to phap ly cua ca nhan", "so giay to chung thuc ca nhan",
                   "so giay to", "so cmnd", "so can cuoc cong dan", "chung minh nhan dan so",
                   "so can cuoc cong dan/chung minh nhan dan", "so giay chung thuc ca nhan"],
    "ngay_cap": ["ngay cap"],
    "noi_cap": ["noi cap"],
    "thuong_tru": ["dia chi thuong tru", "noi dang ky ho khau thuong tru"],
    "lien_lac": ["dia chi lien lac", "cho o hien tai"],
    "phan_von": ["phan von gop", "gia tri phan von gop", "gia tri von gop"],
    "ty_le": ["ty le"],
}
ALIASES = [(a, k) for k, al in LABELS.items() for a in al]
SHORT_LABELS = {"fax", "email", "e-mail"}
JUNK = re.compile(r"scan")                  # scanned / camscanner / cs camscan...
# "5.Người..." / "3.Vấn đề lệ" (OCR sai chữ vẫn là đầu mục), không phải "03...đường";
# dòng rác "03." giữa mục 5 bị section_spans loại vì số mục phải tăng dần
SECTION = re.compile(r"^\s*(\d{1,2})\s*\.(?!\.)\s*(\D.*)")
# 'TÊN 2. Địa chỉ' / 'APPLE 2.Dịa điểm': số mục + chữ hoa + chữ thường giữa dòng
INLINE_SECTION = re.compile(r"(?<=\S)\s+(?=\d{1,2}\s*\.\s*[A-ZĐÀ-Ỹ][a-zà-ỹ])")


def match_label(key):
    """Phần trước ':' -> tên trường, so gần đúng (OCR sai 1-2 ký tự)."""
    k = re.sub(r"^[\W\d_]+", "", norm(key))            # bỏ '- ', '3.' ở đầu
    best, score = None, 0.0
    for alias, field in ALIASES:
        r = SequenceMatcher(None, k, alias).ratio()
        if r > score:
            best, score = field, r
    return best if score >= 0.82 else None


def split_label(line):
    """'Việt Nam Địa chỉ liên lạc: ...' -> ['Việt Nam', 'Địa chỉ liên lạc: ...']."""
    n, pos = norm_map(line)
    for alias, _ in ALIASES:
        m = re.search(rf"(?<=\S )({re.escape(alias)})\s*:", n)
        # nhãn ngắn dễ khớp nhầm giữa câu; chỉ cho các nhãn liên lạc 'Điện thoại: ... Fax: ...'
        if m and (len(alias) > 6 or alias in SHORT_LABELS) \
                and not re.fullmatch(r"[\W\d_]*", n[:m.start(1)]):
            cut = pos[m.start(1)]                       # (không tách số mục '2. ' khỏi nhãn)
            return split_label(line[:cut].rstrip()) + split_label(line[cut:])
    return [line]


def is_label(line):
    key, sep, _ = line.partition(":")
    return bool(sep) and match_label(key) is not None


# ----------------------------------------------------------------- đọc text.txt
def read_text(path):
    """-> (lines, tables). Bảng = khối dòng liền nhau bắt đầu bằng '|'
    (vẫn đọc được text.txt cũ có '--- bảng k' / '---')."""
    if path.lower().endswith((".pdf", ".png", ".jpg", ".jpeg")):
        raise SystemExit(f"{path}: cần file text.txt (kết quả OCR), không phải ảnh/PDF. "
                         "Chạy main.py trước.")
    # utf-8-sig: bỏ BOM; errors="replace": file lẫn ký tự hỏng vẫn đọc được
    with open(path, encoding="utf-8-sig", errors="replace") as f:
        return parse_text(f)


def parse_text(raw_lines):
    """Dòng văn bản OCR -> (lines, tables)."""
    lines, tables, table = [], [], None
    for raw in raw_lines:
        raw = raw.rstrip("\n").split("\t", 1)[-1] if "\t" in raw[:6] else raw.rstrip("\n")
        if JUNK.search(norm(raw)):          # 'Scanned with CamScanner'...: bỏ, không ngắt bảng
            continue
        if raw.startswith("|"):
            cells = [re.sub(r"\s+", " ", re.sub(r"<br\s*/?>", " ", c)).strip()
                     for c in raw.strip().strip("|").split("|")]
            if not all(re.fullmatch(r":?-+:?", c.replace(" ", "")) for c in cells):  # |---|
                table = [] if table is None else table
                table.append(cells)
            continue
        if table is not None:
            tables.append(table)
            table = None
        if raw.startswith("--- bảng") or raw == "---":
            continue
        for part in (p for seg in INLINE_SECTION.split(clean(raw)) for p in split_label(seg)):
            if part:
                lines.append(part)
    if table is not None:
        tables.append(table)
    return join_lines(lines), tables


def join_lines(lines, wrap_len=40):
    """Nối dòng tiếp nối vào dòng có nhãn:
    - nhãn kết thúc bằng ':' (giá trị nằm ở dòng sau);
    - dòng dài bị xuống dòng (>= wrap_len ký tự), dòng sau không phải nhãn / mục mới."""
    out = []
    for line in lines:
        prev = out[-1] if out else ""
        cont = (prev and not is_label(line) and not SECTION.match(line) and ":" in prev
                and not (re.match(r"\s*\d{1,2}\s*\.", line)          # "3..." không phải phần
                         and re.match(r"(?i)\s*(fax|e-?mail|web)", prev))  # sau của fax/email/web
                and (prev.endswith(":") or len(prev) >= wrap_len)
                and (not prev.endswith(".") or norm(line) == "viet nam"))
        if cont:
            out[-1] = f"{prev} {line}"
        else:
            out.append(line)
    return out


def fields(lines):
    """-> list (tên trường, giá trị, chỉ số dòng) của mọi dòng có nhãn."""
    out = []
    for i, line in enumerate(lines):
        key, sep, val = line.partition(":")
        f = match_label(key) if sep else None
        if f:
            out.append((f, val.strip(" ,;.") or None, i))
    return out


def first(fs, name):
    return next((v for f, v, _ in fs if f == name and v), None)


# ----------------------------------------------------------------- chuẩn hóa giá trị
def to_date(s):
    """'13/10/2014' hoặc 'ngày 13 tháng 10 năm 2014' -> '2014-10-13'."""
    if not s:
        return None
    m = re.search(r"(\d{1,2})\s*[/.-]\s*(\d{1,2})\s*[/.-]\s*(\d{4})", s) or \
        re.search(r"ngay\s*(\d{1,2})\s*thang\s*(\d{1,2})\s*nam\s*(\d{4})", norm(s))
    if not m:                                        # OCR mất ngày: '02/1978' -> '1978-02'
        m2 = re.fullmatch(r"\D*(\d{1,2})\s*/\s*(\d{4})\D*", s)
        return f"{m2[2]}-{int(m2[1]):02d}" if m2 and 1 <= int(m2[1]) <= 12 else None
    if not (1 <= int(m[1]) <= 31 and 1 <= int(m[2]) <= 12):
        return None
    return f"{m[3]}-{int(m[2]):02d}-{int(m[1]):02d}"


def to_money(s):
    """'5.000.000.000 đồng' -> 5000000000."""
    m = re.search(r"\d{1,3}(?:[.,\s]\d{3})+|\d{4,}", s or "")
    return int(re.sub(r"\D", "", m[0])) if m else None


def to_percent(s):
    """Kiểu Việt: '40,000' / '40%' -> 40.0 (dấu phẩy là phần thập phân)."""
    m = re.fullmatch(r"(\d{1,3})(?:[,.](\d+))?\s*%?", (s or "").strip())
    return float(f"{m[1]}.{m[2] or 0}") if m and float(f"{m[1]}.{m[2] or 0}") <= 100 else None


def digits(s, n=(9, 13)):
    m = re.search(r"\d[\d\s.]{7,}\d", s or "")
    d = re.sub(r"\D", "", m[0]) if m else ""
    return d if n[0] <= len(d) <= n[1] else None


def phones(s):
    """'0944493499. 0944138610' / '0918 433 830' -> ['0944493499', '0944138610']."""
    out = []
    for part in re.split(r"[,;/]|\.\s|\s-\s", s or ""):
        d = re.sub(r"\D", "", part)
        if 7 <= len(d) <= 12:                       # giấy cũ: số 7 chữ số chưa có mã vùng
            out.append(d)
    return out


def upper_if_mostly(s):
    """'CôNG TY' -> 'CÔNG TY' (OCR lẫn hoa/thường)."""
    if not s:
        return s
    letters = [c for c in s if c.isalpha()]
    return s.upper() if sum(c.isupper() for c in letters) > 0.6 * max(1, len(letters)) else s


def place(s):
    """Địa chỉ: sửa dấu đơn vị hành chính / địa danh, 'Việt nam' -> 'Việt Nam'."""
    s = fix_words(s, "admin")
    if not s:
        return s
    m = re.search(r"\bViệt Nam\b", s, re.I)        # địa chỉ kết thúc ở 'Việt Nam':
    if m:                                             # bỏ chữ ký / con dấu bị nối vào sau
        segs = [g.strip(" .") for g in s[:m.start()].split(",") if g.strip(" .")]
        # sau cấp tỉnh ('Thành phố ...', 'Tỉnh ...') cuối cùng chỉ còn 'Việt Nam': phần thừa là rác
        last = max((i for i, g in enumerate(segs) if norm(g).startswith(("thanh pho", "tinh "))),
                   default=len(segs) - 1)
        s = ", ".join(segs[:last + 1] + ["Việt Nam"])
    return re.sub(r"\b(\w+) \1\b", r"\1", s)       # 'Hồ Chí Minh Minh' -> 'Hồ Chí Minh'


def text(s, kind="admin"):
    return fix_words(s, kind) if s else None


# nơi cấp hay gặp: OCR sai vài chữ vẫn đưa về đúng câu chuẩn
ISSUERS = ["Cục Cảnh sát quản lý hành chính về trật tự xã hội",
           "Cục Cảnh sát đăng ký quản lý cư trú và dữ liệu quốc gia về dân cư",
           "Bộ Công an", "Cục Quản lý xuất nhập cảnh"]


def issuer(s):
    if not s:
        return None
    best = max(ISSUERS, key=lambda c: SequenceMatcher(None, norm(s), norm(c)).ratio())
    return best if SequenceMatcher(None, norm(s), norm(best)).ratio() >= 0.8 else text(s)


def money_words(s):
    s = fix_words(s, "money") if s else s
    return s[:1].upper() + s[1:] if s else None


def email(s):
    m = re.search(r"[\w.+-]+@[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}\b", (s or "").replace(" ", ""))
    return m[0].lower() if m else None


def website(s):
    # đuôi tên miền phải là chữ không dấu (.vn, .com.vn): "3.Vấnđềlệ" không phải website
    m = re.search(r"(?:https?://)?(?:www\.)?[\w-]+(?:\.[\w-]+)*\.[a-zA-Z]{2,}\b",
                  (s or "").replace(" ", ""))
    return m[0].lower() if m else None


# ----------------------------------------------------------------- người
def section_spans(lines):
    """-> [(loại, dòng đầu, dòng cuối)] cho mục thành viên/chủ sở hữu và người đại diện."""
    heads, last = [], 0
    for i, l in enumerate(lines):
        m = SECTION.match(l)
        if m and int(m[1]) > last:              # số mục phải tăng dần: "3." sau "5." là rác
            heads.append((i, norm(l)))
            last = int(m[1])
    spans = []
    for j, (i, h) in enumerate(heads):
        end = heads[j + 1][0] if j + 1 < len(heads) else len(lines)
        if "dai dien" in h:
            spans.append(("dai_dien", i, end))
        elif any(k in h for k in ("chu so huu", "thanh vien", "co dong", "chu doanh nghiep")):
            spans.append(("thanh_vien", i, end))
    return spans


def people(fs, lo, hi):
    """Gom nhãn trong khoảng dòng [lo, hi) thành từng người (mỗi 'Họ và tên' = 1 người)."""
    out = []
    for f, v, i in fs:
        if not lo <= i < hi:
            continue
        if f == "ho_ten":
            out.append({})
        if out and v and f not in out[-1]:
            out[-1][f] = v
    return out


def person(p, rep):
    d = {
        "ho_ten": upper_if_mostly(p.get("ho_ten")),
        "gioi_tinh": text(p.get("gioi_tinh")),
        "ngay_sinh": to_date(p.get("ngay_sinh")),
        "dan_toc": text(p.get("dan_toc")),
        "quoc_tich": text(p.get("quoc_tich")),
        "loai_giay_to": text(p.get("loai_giay_to")),
        "so_giay_to": digits(p.get("so_giay_to")),
        "ngay_cap": to_date(p.get("ngay_cap")),
        "noi_cap": issuer(p.get("noi_cap")),
        "dia_chi_lien_lac": place(p.get("lien_lac")),
    }
    if rep:
        d.update(chuc_danh=text(p.get("chuc_danh")), dia_chi_thuong_tru=place(p.get("thuong_tru")))
    else:
        d.update(dia_chi=place(p.get("thuong_tru")), phan_von_gop=to_money(p.get("phan_von")),
                 ty_le=to_percent(p.get("ty_le")))
    return d


MONEY_RE = re.compile(r"^\d{1,3}(?:[.,]\d{3}){2,}$")
ID_RE = re.compile(r"^\d{9,13}$")
ADDR_RE = re.compile(r"\b(xa|phuong|huyen|quan|tinh|thanh pho|thon|ap|khu pho|viet nam)\b")
NATIONS = {norm(n): n for n in ("Việt Nam", "Trung Quốc", "Hàn Quốc", "Nhật Bản", "Đài Loan",
                                "Singapore", "Hoa Kỳ", "Mỹ", "Pháp", "Đức", "Anh", "Úc")}


def is_name(c):
    """Ô họ tên: in hoa, không số, không dấu phẩy, ngắn. Xét trước địa chỉ vì tên riêng
    có thể trùng từ địa chỉ ('PHAN THỊ BÍCH PHƯƠNG' chứa 'phuong')."""
    return (c.upper() == c and not re.search(r"[\d,]", c)
            and 2 <= len(c.split()) <= 6 and norm(c) not in NATIONS)


def table_members(tables):
    """Bảng thành viên / cổ đông: nhận cột theo nội dung (tiêu đề OCR hay lệch ô)."""
    members, cont = [], False
    for table in tables:
        head = norm(" ".join(table[0]))
        if any(k in head for k in ("thanh vien", "co dong", "ho ten", "ten")):
            rows, cont = table[1:], True
        elif cont and table[0] and re.fullmatch(r"\d{1,3}", table[0][0].strip()):
            rows = table                                    # phần tiếp của bảng bị ngắt trang
        else:
            cont = False
            continue
        for row in rows:
            if not any(re.search(r"\d", c) for c in row):  # hàng rác (dấu 'đã đối chiếu'...)
                continue
            if re.fullmatch(r"\d{1,3}", row[0].strip()) or not members:
                m, cells = {}, row[1:]                      # cột 0 = STT -> thành viên mới
            else:                                           # hàng bị OCR ngắt: nối vào người trước
                m, cells = members.pop(), row
            for c in (clean(c).strip("*_ ") for c in cells):
                if not c:
                    continue
                if MONEY_RE.match(c) and "phan_von_gop" not in m:
                    m["phan_von_gop"] = to_money(c)
                elif norm(c) in NATIONS and "quoc_tich" not in m:
                    m["quoc_tich"] = NATIONS[norm(c)]
                elif ID_RE.match(c.replace(" ", "")) and "so_giay_to" not in m:
                    m["so_giay_to"] = c.replace(" ", "")
                elif to_percent(c) is not None and "ty_le" not in m:
                    m["ty_le"] = to_percent(c)
                elif "ho_ten" not in m and is_name(c):
                    m["ho_ten"] = c
                elif (re.search(r"\d", c) or ADDR_RE.search(norm(c))) and "dia_chi" not in m:
                    m["dia_chi"] = place(c)
                elif "ho_ten" not in m and c.upper() == c:
                    m["ho_ten"] = c
                else:
                    m["ghi_chu"] = c
            if m.get("ho_ten") or m.get("phan_von_gop"):
                members.append(m)
    return members


# ----------------------------------------------------------------- trích xuất
def extract(lines, tables):
    fs = fields(lines)
    n = [norm(l) for l in lines]
    title = next((i for i, l in enumerate(n) if l.startswith("giay chung nhan dang ky")), None)

    # cơ quan cấp: khối tiêu đề (Sở ..., Phòng ĐKKD, tỉnh/thành) - OCR hay đọc 'Sở' thành 'Số'
    head = n[:(title or 0) + 3] if title is not None else n[:8]
    so = next((AGENCIES[a] for l in head for a in AGENCIES if a.startswith("so") and a in l), None)
    phong = next((AGENCIES[a] for l in head for a in AGENCIES if a.startswith("phong") and a in l),
                 None)
    tinh = None
    for l, raw in zip(head, lines):
        m = re.search(r"\b(thanh pho|tinh)\b", l)
        if m and "viet nam" not in l and "doc lap" not in l:
            _, pos = norm_map(raw)
            tinh = raw[pos[m.start()]:].strip()
            tinh = place(tinh.lower() if upper_if_mostly(tinh) == tinh.upper() else tinh)
            tinh = tinh[:1].upper() + tinh[1:]
            break

    thay_doi = first(fs, "thay_doi") or next(
        (l.split("thứ", 1)[-1] for l, k in zip(lines, n) if "dang ky thay doi" in k), None)
    lan = re.match(r"\s*(\d+)", thay_doi or "")
    von = first(fs, "von") or next((lines[i + 1] for i, l in enumerate(n)
                                    if "von dieu le" in l and i + 1 < len(lines)), None)
    if not to_money(von):                               # '3.Vấn đề lệ: 1.000.000.000 đồng'
        von = next((l for l, k in zip(lines, n) if SECTION.match(l) and "dong" in k
                    and (to_money(l) or 0) >= 1_000_000), None)

    members, reps = [], []
    spans = section_spans(lines)
    for kind, lo, hi in spans:
        group = [person(p, kind == "dai_dien") for p in people(fs, lo, hi)]
        (reps if kind == "dai_dien" else members).extend(group)
    if not spans:                                        # không thấy tiêu đề mục: coi là đại diện
        reps = [person(p, True) for p in people(fs, 0, len(lines))]
    members += table_members(tables)

    addr = None
    for i, l in enumerate(lines):                        # 'n. Địa chỉ trụ sở chính' + các dòng sau
        if SECTION.match(l) and "dia chi tru so" in norm(l):
            parts = []
            for nxt in lines[i + 1:]:
                if is_label(nxt) or SECTION.match(nxt):
                    break
                parts.append(nxt)
            addr = place(" ".join(parts)) or None
            break

    return {
        "loai_giay_to": "Giấy chứng nhận đăng ký doanh nghiệp" if title is not None else None,
        "co_quan_cap": {"so": so, "phong": phong, "tinh_thanh": tinh},
        # giữ đúng số OCR đọc được kể cả khi thiếu/thừa 1 chữ số (MSDN chuẩn: 10 hoặc 13)
        "ma_so_doanh_nghiep": digits(first(fs, "ma_so"), (9, 14)),
        "dang_ky_lan_dau": to_date(first(fs, "lan_dau")),
        "dang_ky_thay_doi": {"lan_thu": int(lan[1]) if lan else None, "ngay": to_date(thay_doi)},
        "ten_cong_ty": {
            "tieng_viet": text(upper_if_mostly(first(fs, "ten_vn")), "company"),
            "tieng_nuoc_ngoai": upper_if_mostly(first(fs, "ten_nn")),
            "viet_tat": text(upper_if_mostly(first(fs, "ten_vt")), "company"),
        },
        "tru_so_chinh": {
            "dia_chi": addr,
            "dien_thoai": phones(first(fs, "dien_thoai")),
            "fax": (phones(first(fs, "fax")) or [None])[0],
            "email": email(first(fs, "email")),
            "website": website(first(fs, "website")),
        },
        "von_dieu_le": to_money(von),
        "thanh_vien": members,
        "nguoi_dai_dien": reps,
    }


# ----------------------------------------------------------------- hộ kinh doanh
def section_text(lines, *keys):
    """Giá trị của mục 'n. <nhãn>': phần sau ':' cùng dòng + các dòng sau đến nhãn/mục kế."""
    for i, l in enumerate(lines):
        if SECTION.match(l) and any(k in norm(l) for k in keys):
            parts = [l.partition(":")[2].strip()]
            for nxt in lines[i + 1:]:
                if is_label(nxt) or SECTION.match(nxt):
                    break
                parts.append(nxt)
            return " ".join(p for p in parts if p) or None
    return None


CODE_RE = re.compile(r"\b(\d{4,5})\b")          # mã ngành VSIC 4-5 chữ số
MAIN_RE = re.compile(r"\(?\s*chinh\s*\)?")


def main_business(lines, tables):
    """Ngành nghề chính: dòng/hàng có chữ '(Chính)' hoặc đánh dấu ở cột 'ngành chính'.
    Không có đánh dấu mà chỉ có 1 ngành -> lấy ngành đó; nhiều ngành không đánh dấu -> null."""
    cands = []                                     # (là ngành chính?, tên, mã)
    for table in tables:
        head = [norm(c) for c in table[0]]
        if not any("nganh" in h for h in head):
            continue
        i_ten = next((i for i, h in enumerate(head) if "ten" in h or "nganh nghe" in h), 1)
        i_ma = next((i for i, h in enumerate(head) if h.startswith("ma")), None)
        i_chinh = next((i for i, h in enumerate(head) if "chinh" in h), None)
        for row in table[1:]:
            cells = [clean(c) for c in row]
            if not any(cells):
                continue
            ten = cells[i_ten] if i_ten < len(cells) else ""
            mark = i_chinh is not None and i_chinh < len(cells) and cells[i_chinh].strip(" -")
            is_main = bool(mark) or bool(re.search(r"\(\s*chinh\s*\)", norm(" ".join(cells))))
            code = cells[i_ma] if i_ma is not None and i_ma < len(cells) else " ".join(cells)
            cands.append((is_main, ten, code))
    if not cands:                                  # không có bảng: liệt kê thành dòng trong mục
        start = next((i for i, l in enumerate(lines)
                      if SECTION.match(l) and "nganh" in norm(l)), None)
        rest = [lines[start].partition(":")[2]] + lines[start + 1:] if start is not None else []
        for j, l in enumerate(rest):
            if j and (SECTION.match(l) or is_label(l)):
                break
            # bỏ dòng ghi chú '(Ngành nghề kinh doanh có điều kiện ...)'
            if l.strip() and not norm(l).startswith(("chi tiet", "(")):
                cands.append((bool(re.search(r"\(\s*chinh\s*\)", norm(l))), l, l))
    main = [c for c in cands if c[0]] or (cands if len(cands) == 1 else [])
    if not main:
        return {"ten": None, "ma_nganh": None}
    _, ten, code = main[0]
    m = CODE_RE.search(code or "")
    ten = re.sub(r"\(\s*ch[íi]nh\s*\)|\(\s*\d{4,5}\s*\)", "", ten, flags=re.I)
    ten = re.sub(r"^[\s\-+*\d.]+", "", CODE_RE.sub("", ten) if not m or code == ten else ten)
    return {"ten": text(ten.strip(" -;:.,")) or None, "ma_nganh": m[1] if m else None}


def household_members(tables):
    """Bảng thành viên hộ gia đình: nhận cột theo nội dung như bảng thành viên công ty."""
    out = []
    for table in tables:
        head = norm(" ".join(table[0]))
        if "nganh" in head or not any(k in head for k in ("thanh vien", "ho ten", "ho va ten")):
            continue
        for row in table[1:]:
            m = {}
            for c in (clean(c).strip("*_ ") for c in row[1:]):
                if not c:
                    continue
                if to_date(c) and "ngay_sinh" not in m:
                    m["ngay_sinh"] = to_date(c)
                elif norm(c) in ("nam", "nu") and "gioi_tinh" not in m:
                    m["gioi_tinh"] = "Nam" if norm(c) == "nam" else "Nữ"
                elif norm(c) in NATIONS and "quoc_tich" not in m:
                    m["quoc_tich"] = NATIONS[norm(c)]
                elif ID_RE.match(c.replace(" ", "")) and "so_giay_to" not in m:
                    m["so_giay_to"] = c.replace(" ", "")
                elif "ho_ten" not in m and is_name(c):
                    m["ho_ten"] = c
                elif (re.search(r"\d", c) or ADDR_RE.search(norm(c))) and "dia_chi" not in m:
                    m["dia_chi"] = place(c)
                elif "ho_ten" not in m and c.upper() == c:
                    m["ho_ten"] = c
            if m.get("ho_ten"):
                out.append(m)
    return out


def extract_hkd(lines, tables):
    fs = fields(lines)
    n = [norm(l) for l in lines]
    title = hkd_title(n) or 0
    head = list(zip(n[:title + 1], lines))
    ubnd = phong = None
    for l, r in head:
        _, pos = norm_map(r)
        u = re.search(r"\b(uy ban nhan dan|ubnd)\b", l)
        p = re.search(r"\bphong\b", l)
        if u and not ubnd:
            ubnd = r[pos[u.start()]:pos[p.start()] if p and p.start() > u.start() else None]
        if p and not phong:
            phong = r[pos[p.start()]:]
    fmt_cap = lambda s: fix_words(re.sub(r"\s*-\s*", " - ", s.strip(" ,."))) if s else None

    thay_doi = first(fs, "thay_doi") or next(
        (l.split("thứ", 1)[-1] for l, k in zip(lines, n) if "dang ky thay doi" in k), None)
    lan = re.match(r"\s*(\d+)", thay_doi or "")
    def code(s):                     # '8005620026', mẫu cũ có chữ: '53K8003560', '53KHB004296'
        m = re.search(r"[0-9A-Z][0-9A-Z\s]{7,}[0-9]", (s or "").upper())
        return re.sub(r"\s", "", m[0]) if m else None
    ma = code(first(fs, "ma_so_hkd") or first(fs, "ma_so"))
    so_dk = code(first(fs, "so_dk_hkd"))
    if not so_dk:                                      # mẫu cũ: 'Số: 53KHB004296' ở đầu giấy
        so_dk = next((code(r.partition(":")[2]) for l, r in zip(n[:title + 3], lines)
                      if re.match(r"so\s*:", l)), None)

    def reg_date(label):             # 'Đăng ký lần đầu, ngày 5 tháng 6 năm 2020' (không có ':')
        return to_date(first(fs, label)) or next(
            (to_date(l) for l, k in zip(lines, n) if LABELS[label][0] in k), None)

    owner_lo = next((i for i, l in enumerate(n) if SECTION.match(lines[i])
                     and ("chu ho" in l or "dai dien ho kinh doanh" in l)), None)
    owner_hi = next((i for i, l in enumerate(n) if owner_lo is not None and i > owner_lo
                     and SECTION.match(lines[i])), len(lines))
    owners = people(fs, owner_lo, owner_hi) if owner_lo is not None else people(fs, 0, len(lines))
    # Chủ hộ chỉ có một người; OCR có thể lặp nhãn 'Họ và tên' giữa mục -> gộp phần tiếp theo.
    owner = dict(owners[0]) if owners else {}
    for extra in owners[1:]:
        for key, value in extra.items():
            owner.setdefault(key, value)
    chu = person(owner, True) if owner else {}
    chu.pop("chuc_danh", None)

    von = first(fs, "von_kd") or section_text(lines, "von kinh doanh")
    return {
        "loai_giay_to": "Giấy chứng nhận đăng ký hộ kinh doanh",
        "co_quan_cap": {"ubnd": fmt_cap(ubnd), "phong": fmt_cap(phong)},
        "ma_so_ho_kinh_doanh": ma,
        "so_dang_ky": so_dk,
        "dang_ky_lan_dau": reg_date("lan_dau"),
        "dang_ky_thay_doi": {"lan_thu": int(lan[1]) if lan else None, "ngay": to_date(thay_doi)},
        "ten_ho_kinh_doanh": hkd_name(first(fs, "ten_hkd") or section_text(lines, "ten ho kinh doanh")),
        "dia_diem_kinh_doanh": {
            "dia_chi": place(first(fs, "dia_diem") or section_text(lines, "dia diem", "dia chi")),
            "dien_thoai": phones(first(fs, "dien_thoai")),
            "fax": (phones(first(fs, "fax")) or [None])[0],
            "email": email(first(fs, "email")),
            "website": website(first(fs, "website")),
        },
        "nganh_nghe_chinh": main_business(lines, tables),
        "von_kinh_doanh": to_money(von),
        "chu_ho": chu,
        "thanh_vien_ho": household_members(tables),
    }


def hkd_name(s):
    """'HQ KINH DOANH ...' (OCR đọc sai 'HỘ') -> 'HỘ KINH DOANH ...'."""
    s = text(upper_if_mostly(s), "company")
    s = s.rstrip(" .") if s else s
    return re.sub(r"^H\S{0,2}\s+KINH DOANH\b", "HỘ KINH DOANH", s, flags=re.I) if s else None


def hkd_title(n):
    """Chỉ số dòng tiêu đề GCN hộ kinh doanh; tiêu đề có thể bị OCR tách 2 dòng
    ('GIẤY CHỨNG NHẬN ĐĂNG KÝ' / 'HỘ KINH DOANH'). Không thấy -> None."""
    for i, l in enumerate(n[:15]):
        # OCR hay đọc sai 'HỘ' thành 'HỢP', 'HO', 'HQ'...
        if re.search(r"dang ky h\w{0,3} kinh doanh", l):
            return i
        if l.endswith("dang ky") and i + 1 < len(n) and re.match(r"h\w{0,3} kinh doanh", n[i + 1]):
            return i + 1
    return None


def is_hkd(lines):
    return hkd_title([norm(l) for l in lines]) is not None


def conform(data, tpl):
    """Ép về đúng khóa của mẫu: thiếu -> null/[], thừa -> bỏ."""
    if isinstance(tpl, dict):
        data = data if isinstance(data, dict) else {}
        return {k: conform(data.get(k), v) for k, v in tpl.items()}
    if isinstance(tpl, list):
        if not isinstance(data, list):
            return []
        return [conform(d, tpl[0]) for d in data] if tpl and isinstance(tpl[0], dict) else data
    return data


def load_templates(dn=TEMPLATE, hkd=TEMPLATE_HKD):
    tpls = {}
    for key, path in (("dn", dn), ("hkd", hkd)):
        with open(path, encoding="utf-8") as f:
            tpls[key] = json.load(f)
    return tpls


CCCD, CMND = "Căn cước công dân", "Chứng minh nhân dân"


def id_type(kind, number):
    """Loại giấy tờ cá nhân chỉ là CCCD hoặc CMND. OCR sai ('7', rác) -> đoán theo số:
    12 chữ số = CCCD, 9 chữ số = CMND; không rõ -> None."""
    k = norm(kind or "").replace(" ", "")
    if "cancuoc" in k or "cccd" in k or "cuoc" in k:
        return CCCD
    if "chungminh" in k or "cmnd" in k or "minhnhan" in k:
        return CMND
    n = re.sub(r"\D", "", number or "")
    return CCCD if len(n) == 12 else CMND if len(n) == 9 else None


def fix_id_types(d):
    """Chuẩn hóa loai_giay_to của mọi người trong kết quả (thành viên, đại diện, chủ hộ...)."""
    for k, v in d.items():
        if isinstance(v, dict):
            if "so_giay_to" in v:
                v["loai_giay_to"] = id_type(v.get("loai_giay_to"), v.get("so_giay_to"))
            fix_id_types(v)
        elif isinstance(v, list):
            for x in v:
                if isinstance(x, dict):
                    fix_id_types({"_": x})
    return d


def extract_text(text, tpls=None, kind=None):
    """Văn bản OCR -> dict theo mẫu. kind: "dn"/"hkd"; None thì tự nhận loại giấy."""
    tpls = tpls or load_templates()
    lines, tables = parse_text(unicodedata.normalize("NFC", text).splitlines())
    if kind == "hkd" or kind is None and is_hkd(lines):
        return fix_id_types(conform(extract_hkd(lines, tables), tpls["hkd"]))
    return fix_id_types(conform(extract(lines, tables), tpls["dn"]))


def extract_file(path, tpls=None):
    """text.txt -> dict theo mẫu, tự nhận loại giấy (doanh nghiệp / hộ kinh doanh)."""
    tpls = tpls or load_templates()
    lines, tables = read_text(path)
    if is_hkd(lines):
        return fix_id_types(conform(extract_hkd(lines, tables), tpls["hkd"]))
    return fix_id_types(conform(extract(lines, tables), tpls["dn"]))


def main():
    ap = argparse.ArgumentParser(description="Trích xuất GCN đăng ký doanh nghiệp từ text.txt")
    ap.add_argument("inputs", nargs="+", help="file text.txt hoặc thư mục chứa */text.txt")
    ap.add_argument("-o", "--out", help="file .json khi chỉ có 1 đầu vào (mặc định: in ra)")
    ap.add_argument("--template", default=TEMPLATE, help="mẫu JSON cho doanh nghiệp")
    args = ap.parse_args()

    tpls = load_templates(args.template)
    paths = []
    for p in args.inputs:
        if os.path.isdir(p):
            paths += sorted(os.path.join(d, "text.txt") for d, _, fs in os.walk(p) if "text.txt" in fs)
        else:
            paths.append(p)
    if not paths:
        raise SystemExit("Không tìm thấy text.txt")

    for path in paths:
        info = json.dumps(extract_file(path, tpls), ensure_ascii=False, indent=2)
        if len(paths) == 1 and not args.out and not os.path.isdir(args.inputs[0]):
            print(info)
            continue
        out = args.out if len(paths) == 1 and args.out else \
            os.path.join(os.path.dirname(path), "extract.json")
        with open(out, "w", encoding="utf-8") as f:
            f.write(info + "\n")
        print(f"-> {out}")


if __name__ == "__main__":
    main()
