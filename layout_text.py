"""Dựng text OCR thành dạng giữ bố cục: đặt từng vùng vào lưới ký tự theo tọa độ box."""


def render_layout(regions, page_size, cols=160, char_aspect=2.0):
    """regions: [{"box": [x1, y1, x2, y2], "text": str}], page_size: (width, height) pixel.

    Mỗi ký tự rộng page_width/cols pixel, cao gấp char_aspect lần (font monospace).
    Vùng đè lên chữ đã đặt sẽ được đẩy xuống dòng trống kế tiếp, không ghi đè.
    """
    width = max(1, page_size[0])
    char_w = width / cols
    line_h = char_w * char_aspect
    grid = []

    def free(row, col, length):
        if row >= len(grid):
            return True
        line = grid[row]
        return all(c == " " for c in line[col:col + length])

    def put(row, col, text):
        while len(grid) <= row:
            grid.append([])
        line = grid[row]
        if len(line) < col + len(text):
            line.extend(" " * (col + len(text) - len(line)))
        line[col:col + len(text)] = text

    for region in sorted(regions, key=lambda r: (r["box"][1], r["box"][0])):
        lines = [line.rstrip() for line in region["text"].strip("\n").splitlines()] or [""]
        x1, y1 = region["box"][:2]
        col, row = max(0, round(x1 / char_w)), max(0, round(y1 / line_h))
        # Chừa 1 ô trống bên trái để hai vùng sát nhau không dính chữ.
        while not all(free(row + i, max(0, col - 1), len(line) + 1) for i, line in enumerate(lines)):
            row += 1
        for i, line in enumerate(lines):
            put(row + i, col, line)
    return "\n".join("".join(line).rstrip() for line in grid)
