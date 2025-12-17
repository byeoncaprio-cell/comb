import tkinter as tk
from tkinter import ttk, messagebox, filedialog, colorchooser, simpledialog
from typing import Set, Tuple, List, Dict, Union, Optional
from dataclasses import dataclass, field
from collections import Counter, OrderedDict
import random
import copy
import os
import re
import sys
import pickle  # 프로젝트 저장/불러오기를 위해 추가

# 엑셀 관련
from openpyxl import Workbook, load_workbook
from openpyxl.styles import PatternFill, Alignment, Border, Side, Font
from openpyxl.utils import get_column_letter
from openpyxl.cell import Cell

# PDF 및 OCR 관련 라이브러리
import fitz  # PyMuPDF
import numpy as np
import pandas as pd
import cv2
import pytesseract
from pytesseract import Output

# (선택) 환경변수로 Tesseract 경로 지정 가능
if os.environ.get("TESSERACT_CMD"):
    pytesseract.pytesseract.tesseract_cmd = os.environ["TESSERACT_CMD"]

def is_valid_pdf(path: str) -> bool:
    try:
        if not os.path.isfile(path) or os.path.getsize(path) < 10:
            return False
        with open(path, "rb") as f:
            head = f.read(1024)
        return b"%PDF-" in head
    except Exception:
        return False

def extract_words_pdf(page):
    words = page.get_text("words")
    rows = []
    for (x0, y0, x1, y1, text, *_rest) in words:
        rows.append({
            "text": text,
            "x0": x0, "y0": y0, "x1": x1, "y1": y1,
            "cx": (x0 + x1) / 2, "cy": (y0 + y1) / 2
        })
    return pd.DataFrame(rows)

def preprocess_for_ocr(img_bgr):
    gray = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)
    gray = cv2.normalize(gray, None, 0, 255, cv2.NORM_MINMAX)
    thr = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
    return thr
    
def ocr_words(page, zoom=5.0, whitelist="()R0123456789"):
    mat = fitz.Matrix(zoom, zoom)
    pix = page.get_pixmap(matrix=mat, alpha=False)
    img = np.frombuffer(pix.samples, dtype=np.uint8).reshape(pix.h, pix.w, 3).copy()

    binimg = preprocess_for_ocr(img)
    rotation_angle = 0
    try:
        osd_data = pytesseract.image_to_osd(binimg, output_type=Output.DICT)
        rotation_angle = int(osd_data.get('rotate', 0))
    except Exception:
        pass
    
    rotated_binimg = binimg
    
    if rotation_angle == 90:
        rotated_binimg = cv2.rotate(binimg, cv2.ROTATE_90_COUNTERCLOCKWISE)
    elif rotation_angle == 180:
        rotated_binimg = cv2.rotate(binimg, cv2.ROTATE_180)
    elif rotation_angle == 270:
        rotated_binimg = cv2.rotate(binimg, cv2.ROTATE_90_CLOCKWISE)

    config = f"--oem 3 --psm 6 -c tessedit_char_whitelist={whitelist}"
    df = pytesseract.image_to_data(rotated_binimg, output_type=Output.DATAFRAME, config=config)
    
    df = df.dropna()
    if "conf" in df.columns:
        df = df[df["conf"].astype(float) >= 35].copy()
    if df.empty:
        return pd.DataFrame(columns=["text","x0","y0","x1","y1","cx","cy"])
    df["x0"] = df["left"]
    df["y0"] = df["top"]
    df["x1"] = df["left"] + df["width"]
    df["y1"] = df["top"] + df["height"]
    df["cx"] = df["x0"] + df["width"]/2
    df["cy"] = df["y0"] + df["height"]/2

    rotated_pix_h, rotated_pix_w = rotated_binimg.shape[:2]
    x_scale = page.rect.width  / rotated_pix_w
    y_scale = page.rect.height / rotated_pix_h
    
    for col in ["x0","x1","cx"]:
        df[col] = df[col] * x_scale
    for col in ["y0","y1","cy"]:
        df[col] = df[col] * y_scale

    return df[["text","x0","y0","x1","y1","cx","cy"]].copy()

def find_R_with_number(df):
    full_pattern = re.compile(r"\(?[Rr]\s*[-_.]?\s*[0-9]+\)?")
    r_only_pattern = re.compile(r"\(?[Rr]\)?")
    num_only_pattern = re.compile(r"\(?[0-9]+\)?")

    rows = []
    skip_next = False
    
    for i in range(len(df)):
        if skip_next:
            skip_next = False
            continue
            
        t = df.iloc[i]
        txt = str(t["text"]).strip()

        if full_pattern.fullmatch(txt):
            num_only = re.sub(r"\D", "", txt)
            rows.append({
                "token": txt,
                "num": num_only,
                "x": t["cx"], "y": t["cy"],
                "x0": t["x0"], "y0": t["y0"], "x1": t["x1"], "y1": t["y1"]
            })

        elif r_only_pattern.fullmatch(txt):
            if i + 1 < len(df):
                next_t = df.iloc[i+1]
                next_txt = str(next_t["text"]).strip()
                
                if num_only_pattern.fullmatch(next_txt):
                    dist_x = next_t['x0'] - t['x1']
                    char_height = t['y1'] - t['y0']
                    
                    if dist_x < char_height * 3: 
                        combined_txt = txt + next_txt 
                        num_only = re.sub(r"\D", "", next_txt)
                        new_cx = (t['x0'] + next_t['x1']) / 2
                        new_cy = (t['y0'] + next_t['y1']) / 2
                        
                        rows.append({
                            "token": combined_txt, 
                            "num": num_only,
                            "x": new_cx, "y": new_cy,
                            "x0": t["x0"], "y0": t["y0"], 
                            "x1": next_t["x1"], "y1": next_t["y1"]
                        })
                        skip_next = True 
            
    return pd.DataFrame(rows)

def group_tokens_by_x_and_y(tokens, x_tol=20.0, y_tol=15.0, use_adaptive_tol=True, y_k=1.2):
    if tokens.empty:
        return tokens.copy()

    df = tokens.copy().reset_index(drop=True)

    if use_adaptive_tol and {"x0", "x1"}.issubset(df.columns):
        median_w = float(np.median((df["x1"] - df["x0"]).abs()))
        if median_w > 0:
            x_tol = max(x_tol, 1.5 * median_w)

    cols = []
    order = df.sort_values("x", ascending=False).index.tolist()
    for idx in order:
        x = df.at[idx, "x"]
        best_j, best_dx = None, None
        for j, col in enumerate(cols):
            dx = abs(x - col["x"])
            if dx <= x_tol and (best_dx is None or dx < best_dx):
                best_dx, best_j = dx, j
        if best_j is None:
            cols.append({"x": float(x), "idxs": [idx]})
        else:
            col = cols[best_j]
            col["idxs"].append(idx)
            col["x"] = float(np.mean(df.loc[col["idxs"], "x"]))

    cols = sorted(cols, key=lambda c: c["x"], reverse=True)
    col_id_map = {}
    for c_id, col in enumerate(cols, start=1):
        for i in col["idxs"]:
            col_id_map[i] = c_id
    df["ColumnID"] = df.index.map(col_id_map)

    group_id = 0
    group_ids = [None] * len(df)
    for c in cols:
        col_indices = sorted(c["idxs"], key=lambda i: df.at[i, "y"])
        if len(col_indices) >= 2:
            ys = [df.at[i, "y"] for i in col_indices]
            diffs = np.diff(ys)
            diffs = np.abs(diffs)
            med_dy = float(np.median(diffs)) if len(diffs) else 0.0
            split_thr = max(y_tol, y_k * med_dy)
        else:
            split_thr = y_tol
        prev_y = None
        for i in col_indices:
            y = df.at[i, "y"]
            if prev_y is None:
                group_id += 1
            else:
                if abs(y - prev_y) > split_thr:
                    group_id += 1
            group_ids[i] = group_id
            prev_y = y

    df["GroupID"] = group_ids
    df = df.sort_values(by=["ColumnID", "GroupID", "y"], ascending=[True, True, True]).reset_index(drop=True)
    return df

Cell = Tuple[int, int]

@dataclass
class Block:
    rows: int
    cols: int
    hatch: str = ""
    hold: str = ""
    bay: str = ""
    deck: str = ""
    cell_colors: Dict[Cell, str] = field(default_factory=dict)
    cell_numbers: Dict[Cell, Union[int, float, str]] = field(default_factory=dict)
    sockets: Set[Cell] = field(default_factory=set)
    gang_counts: Dict[int, int] = field(default_factory=dict)
    is_hold: bool = field(init=False)

    def __post_init__(self):
        self.is_hold = self.rows >= 6

@dataclass
class SectionHeader:
    title: str

Item = Union[Block, SectionHeader]

SHAPE_LIBRARY: Dict[int, List[List[Cell]]] = {
    9: [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        [(0, 1), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (4, 1)],
        [(0, 0), (1, 0), (1, 1), (1, 2), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (2, 2), (3, 0), (3, 1), (3, 2)],
        [(0, 1), (0, 2), (0, 3), (0, 4), (1, 0), (1, 1), (1, 2), (1, 3), (1, 4)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4), (1, 1), (1, 2), (1, 3), (1, 4)],
        [(0, 1), (1, 1), (2, 1), (3, 0), (4, 0), (5, 0), (3, 1), (4, 1), (5, 1)],
    ],
    8: [
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)], 
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
    ],
    7: [
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 1)],
        [(0, 0), (0, 1), (0, 2), (0, 3), (1, 1), (1, 2), (1, 3)],
        [(0, 1), (0, 2), (0, 3), (1, 0), (1, 1), (1, 2), (1, 3)],
        [(0, 0), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0), (3, 1)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1), (3, 0)],
    ],
    6: [
        [(0, 0), (0, 1), (0, 2), (1, 0), (1, 1), (1, 2)],
        [(0, 0), (0, 1), (1, 0), (1, 1), (2, 0), (2, 1)],
    ],
    5: [
        [(0, 0), (0, 1), (0, 2), (0, 3), (0, 4)],
        [(0, 0), (1, 0), (2, 0), (3, 0), (4, 0)],
        [(0, 0), (0, 1), (1, 1), (2, 1), (3, 1)]
    ],
    4: [
        [(0, 0), (0, 1), (1, 0), (1, 1)],
    ],
}

def parse_number_like(s: str) -> Union[int, float, str]:
    try:
        if str(s).strip() == "": return ""
        if "." in str(s):
            f = float(s)
            return int(f) if f.is_integer() else f
        return int(s)
    except (ValueError, TypeError):
        return s

def header_sequence(n: int) -> List[int]:
    evens = [x for x in range(n, 0, -1) if x % 2 == 0]
    odds = list(range(1, n + 1, 2))
    return evens + odds
    
def set_range_border(ws, min_row, max_row, min_col, max_col, side_top, side_right, side_bottom, side_left):
    for r in range(min_row, max_row + 1):
        for c in range(min_col, max_col + 1):
            cell = ws.cell(row=r, column=c)
            current_border = cell.border
            new_border = Border(
                top=side_top if r == min_row else current_border.top,
                bottom=side_bottom if r == max_row else current_border.bottom,
                left=side_left if c == min_col else current_border.left,
                right=side_right if c == max_col else current_border.right
            )
            cell.border = new_border

def thick_column_positions(max_grid_cols: int) -> List[int]:
    if max_grid_cols <= 0: return []
    seq = header_sequence(max_grid_cols)
    try:
        boundary_col = seq.index(1) + 1
    except ValueError:
        return []
    pos = {x for x in range(boundary_col, max_grid_cols + 1, 4)}
    pos.update({x for x in range(boundary_col - 4, 0, -4)})
    return sorted(list(pos))

def safe_sheet_title(s: str) -> str:
    invalid_chars = r'[]:*?/\ '
    clean_s = "".join(c if c not in invalid_chars else '-' for c in (s or "Sheet"))
    return clean_s[:31]

def _collect_rd_counts(items: List[Item]) -> OrderedDict:
    c = Counter(int(v) for it in items if isinstance(it,Block) for v in it.cell_numbers.values() if str(v).strip())
    return OrderedDict(sorted(c.items()))

def _rd_list_for_rs(rs_index, rd_per_rs):
    if rd_per_rs <= 0 or rs_index <= 0: return []
    if rs_index % 2 == 1:
        block_idx = (rs_index - 1) // 2
        start_odd = 1 + 2 * (block_idx * rd_per_rs)
        return [start_odd + 2*i for i in range(rd_per_rs)]
    else:
        block_idx = (rs_index // 2) - 1
        start_even = 2 + 2 * (block_idx * rd_per_rs)
        return [start_even + 2*i for i in range(rd_per_rs)]

def _build_rd_queues(rs_total, rpr):
    even_q, odd_q = [], []
    for rs in range(1, rs_total + 1):
        (even_q if rs % 2 == 0 else odd_q).extend(_rd_list_for_rs(rs, rpr))
    return sorted(list(set(even_q))), sorted(list(set(odd_q)))

def _build_rs_summary(rs_indices, rd_counts, rpr):
    lines = []
    for rs in rs_indices:
        rd_list = _rd_list_for_rs(rs, rpr)
        total = sum(rd_counts.get(rd, 0) for rd in rd_list)
        lines.append(f"RS-{rs}: total {total}")
        lines.extend([f"  RD-{rd}: {rd_counts[rd]}" for rd in rd_list if rd_counts.get(rd, 0) > 0])
        if not any(rd_counts.get(rd, 0) > 0 for rd in rd_list):
            lines.append("  -")
        lines.append("")
    return "\n".join(lines).rstrip()

def write_excel(
    items: List[Item],
    filename: str,
    ship_no: str,
    looking_text: str = "LOOKING TO FWD",
    rs_count_str: str = "0",
    rd_per_rs_str: str = "0",
    row_gap_between_blocks: int = 2,
    info_cols: int = 1, gap_after_info: int = 1,
    deck_cols: int = 1, gap_after_deck: int = 1
):
    if not items:
        raise ValueError("No content to export. Add blocks or section headers.")

    block_cols = [it.cols for it in items if isinstance(it, Block)]
    max_grid_cols = max(block_cols) if block_cols else 6

    grid_start_col = info_cols + gap_after_info + deck_cols + gap_after_deck + 1
    grid_right_col = grid_start_col + max_grid_cols - 1

    qty_col = grid_right_col + 2
    g_start_col = qty_col + 1
    g_cols = {g: g_start_col + (g - 3) for g in range(3, 10)}
    g_sum_col = g_start_col + 7

    wb = Workbook()
    ws = wb.active    
    sheet_name = safe_sheet_title(ship_no)
    ws.title = sheet_name

    align_center = Alignment(horizontal="center", vertical="center")
    align_right  = Alignment(horizontal="right",  vertical="center")
    thin  = Side(style="thin", color="000000")
    thick = Side(style="thick", color="000000")
    dashed_med = Side(style="mediumDashed", color="000000")
    border_thin    = Border(left=thin, right=thin, top=thin, bottom=thin)
    gray_fill = PatternFill("solid", fgColor="EEEEEE")

    for c in range(1, g_sum_col + 2): ws.column_dimensions[get_column_letter(c)].width = 4
    for c in range(1, info_cols + 1): ws.column_dimensions[get_column_letter(c)].width = 10
    ws.column_dimensions[get_column_letter(info_cols + 1)].width = 2
    for c in range(info_cols + 2, info_cols + 2 + deck_cols): ws.column_dimensions[get_column_letter(c)].width = 4
    ws.column_dimensions[get_column_letter(info_cols + 2 + deck_cols)].width = 2
    
    ws.column_dimensions[get_column_letter(grid_right_col + 1)].width = 2
    ws.column_dimensions[get_column_letter(qty_col)].width = 4
    for g in range(3, 10): ws.column_dimensions[get_column_letter(g_cols[g])].width = 4
    ws.column_dimensions[get_column_letter(g_sum_col)].width = 7

    title_text = (ship_no.strip() + " Reefer Arrangement") if ship_no.strip() else "Reefer Arrangement"
    try:
        ws.oddHeader.center.text = f"&B&11{title_text}"
        ws.oddHeader.right.text  = f"&B&11{(looking_text or 'LOOKING TO FWD')}"
    except Exception:
        pass
    left_end = max(1, grid_right_col - 4)
    ws.merge_cells(start_row=1, start_column=1, end_row=1, end_column=left_end)
    tcell = ws.cell(row=1, column=1, value=title_text)
    tcell.alignment = Alignment(horizontal="center", vertical="center"); tcell.font = Font(size=11, bold=True)
    set_range_border(ws, 1, 1, 1, left_end, thick, thick, thick, thick)

    right_start = left_end + 1
    ws.merge_cells(start_row=1, start_column=right_start, end_row=1, end_column=grid_right_col)
    rcell = ws.cell(row=1, column=right_start, value=(looking_text or "LOOKING TO FWD"))
    rcell.alignment = Alignment(horizontal="center", vertical="center"); rcell.font = Font(size=10)
    set_range_border(ws, 1, 1, right_start, grid_right_col, thick, thick, thick, thick)

    cur_row = 3
    seq = header_sequence(max_grid_cols)
    for i, val in enumerate(seq, start=0):
        cc = grid_start_col + i
        ws.cell(row=cur_row, column=cc, value=int(val)).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=cur_row, column=cc).font = Font(size=9, bold=True)

    ws.merge_cells(start_row=cur_row+1, start_column=g_cols[3], end_row=cur_row+1, end_column=g_sum_col)
    gtitle = ws.cell(row=cur_row+1, column=g_cols[3], value="gang별 수량")
    gtitle.alignment = Alignment(horizontal="center", vertical="center"); gtitle.font = Font(size=9, bold=True)
    for c in range(g_cols[3], g_sum_col + 1):
        ws.cell(row=cur_row+1, column=c).fill = gray_fill

    hrow = cur_row + 2
    ws.cell(row=hrow, column=qty_col, value="Q'ty").alignment = Alignment(horizontal="center", vertical="center")
    for g in range(3, 10):
        hc = ws.cell(row=hrow, column=g_cols[g], value=g)
        hc.alignment = Alignment(horizontal="center", vertical="center"); hc.fill = gray_fill
    sm = ws.cell(row=hrow, column=g_sum_col, value="SUM")
    sm.alignment = Alignment(horizontal="center", vertical="center"); sm.fill = gray_fill

    cur_row = hrow + 1
    data_start_row = cur_row 
    vertical_thick_cols = set(thick_column_positions(max_grid_cols))
    last_used_row = cur_row
    grand_qty_ranges = []
    gang_summary_rows = {'LB': [], 'HOLD': []} 
    count_left_letter  = get_column_letter(grid_start_col)
    count_right_letter = get_column_letter(grid_start_col + max_grid_cols - 1)

    for idx, it in enumerate(items):
        if isinstance(it, SectionHeader):
            ws.merge_cells(start_row=cur_row, start_column=grid_start_col, end_row=cur_row, end_column=grid_right_col)
            scell = ws.cell(row=cur_row, column=grid_start_col, value=it.title)
            scell.alignment = Alignment(horizontal="center", vertical="center"); scell.font = Font(size=24, bold=True)
            set_range_border(ws, cur_row, cur_row, grid_start_col, grid_right_col, dashed_med, dashed_med, dashed_med, dashed_med)
            cur_row += 4
            last_used_row = max(last_used_row, cur_row - 1)
            continue
        b: Block = it
        label_text = f"Hatch No. {b.hatch}" if b.hatch else (f"Hold No. {b.hold}" if b.hold else " ")
        ws.cell(row=cur_row, column=1, value=label_text).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=cur_row, column=1).font = Font(size=8, color="C00000")
        set_range_border(ws, cur_row, cur_row, 1, 1, thin, thin, thin, thin)

        bay_text = f"Bay {b.bay}" if b.bay else " "
        ws.cell(row=cur_row + 1, column=1, value=bay_text).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=cur_row + 1, column=1).font = Font(size=8, color="C00000")
        set_range_border(ws, cur_row + 1, cur_row + 1, 1, 1, thin, thin, thin, thin)
        deck_text = b.deck if b.deck else " "
        ws.cell(row=cur_row, column=info_cols + gap_after_info + 1, value=deck_text).alignment = Alignment(horizontal="center", vertical="center")
        ws.cell(row=cur_row, column=info_cols + gap_after_info + 1).font = Font(size=8, color="C00000")

        left_offset = (max_grid_cols - b.cols) // 2
        block_start_row = cur_row
        block_end_row   = cur_row + b.rows - 1
    
        grand_qty_ranges.append((block_start_row, block_end_row))

        for r in range(1, b.rows + 1):
            rr = cur_row + r - 1
            ws.row_dimensions[rr].height = 18 
            for c in range(1, b.cols + 1):
                global_grid_col = left_offset + c
                cc = grid_start_col + global_grid_col - 1
                cell = ws.cell(row=rr, column=cc)
                cell.alignment = Alignment(horizontal="center", vertical="center")
                if (r,c) in b.cell_colors: cell.fill = PatternFill("solid", fgColor=b.cell_colors[(r,c)].lstrip("#"))
                cell.border = border_thin
                if (r,c) in b.cell_numbers: cell.value = parse_number_like(b.cell_numbers[(r,c)])
                if (r,c) in b.sockets: cell.border = Border(left=Side(style="thick", color="FF0000"), right=Side(style="thick", color="FF0000"), top=Side(style="thick", color="FF0000"), bottom=Side(style="thick", color="FF0000"))
                if global_grid_col in vertical_thick_cols: cell.border = Border(left=thick, right=cell.border.right, top=cell.border.top, bottom=cell.border.bottom)

            ws.cell(row=rr, column=qty_col, value=f"=COUNT({count_left_letter}{rr}:{count_right_letter}{rr})").alignment = Alignment(horizontal="center", vertical="center")
            ws.cell(row=rr, column=qty_col).font = Font(size=8)
            for g in range(3, 10): ws.cell(row=rr, column=g_cols[g]).fill = gray_fill
            ws.cell(row=rr, column=g_sum_col).fill = gray_fill

        sum_row = block_end_row + 1
        block_type = 'HOLD' if b.is_hold else 'LB'
        gang_summary_rows[block_type].append(sum_row)

        qcol_letter = get_column_letter(qty_col)
        total_cell = ws.cell(row=sum_row, column=qty_col, value=f"=SUM({qcol_letter}{block_start_row}:{qcol_letter}{block_end_row})")
        total_cell.alignment = Alignment(horizontal="center", vertical="center"); total_cell.font = Font(size=8, bold=True)
        total_cell.border = Border(top=thick, left=thin, right=thin, bottom=thin)

        terms = []
        for g in range(3, 10):
            cnt = int(b.gang_counts.get(g, 0) or 0)
            cell = ws.cell(row=sum_row, column=g_cols[g], value=cnt)
            cell.alignment = Alignment(horizontal="center", vertical="center"); cell.font = Font(size=8); cell.fill = gray_fill
            terms.append(f"{g}*{get_column_letter(g_cols[g])}{sum_row}")

        gsum_cell = ws.cell(row=sum_row, column=g_sum_col, value="=" + "+".join(terms))
        gsum_cell.alignment = Alignment(horizontal="center", vertical="center"); gsum_cell.font = Font(size=8, bold=True); gsum_cell.fill = gray_fill

        cur_row += max(b.rows, 2) + row_gap_between_blocks
        last_used_row = max(last_used_row, cur_row - 1)
        
    grand_row = last_used_row + 2
    ws.cell(row=grand_row, column=(qty_col - 1), value="Total").alignment = Alignment(horizontal="right", vertical="center"); ws.cell(row=grand_row, column=(qty_col-1)).font = Font(size=9, bold=True)
    parts = [f"{get_column_letter(qty_col)}{s}:{get_column_letter(qty_col)}{e}" for s, e in grand_qty_ranges]
    grand_qty_cell = ws.cell(row=grand_row, column=qty_col, value=f"=SUM({','.join(parts)})")
    grand_qty_cell.alignment = Alignment(horizontal="center", vertical="center"); grand_qty_cell.font = Font(size=9, bold=True)
    grand_qty_cell.border = Border(top=thick, left=thin, right=thin, bottom=thin)
    
    gsum_col_letter = get_column_letter(g_sum_col)
    all_summary_rows = gang_summary_rows['LB'] + gang_summary_rows['HOLD']
    gsum_parts = [f"{gsum_col_letter}{r}" for r in all_summary_rows] if all_summary_rows else ["0"]
    grand_gsum_cell = ws.cell(row=grand_row, column=g_sum_col, value=f"=SUM({','.join(gsum_parts)})")
    grand_gsum_cell.alignment = Alignment(horizontal="center", vertical="center"); grand_gsum_cell.font = Font(size=9, bold=True)
    grand_gsum_cell.border = Border(top=thick, left=thin, right=thin, bottom=thin)
    
    lb_hold_start_row = grand_row + 2
    for i, block_type in enumerate(['LB', 'HOLD']):
        current_row = lb_hold_start_row + i
        ws.cell(row=current_row, column=qty_col, value=block_type).alignment = Alignment(horizontal="right", vertical="center")
        ws.cell(row=current_row, column=qty_col).font = Font(bold=True)
        
        summary_rows_for_type = gang_summary_rows[block_type]        
        for g in range(3, 10):
            col_letter = get_column_letter(g_cols[g])
            sum_parts = [f"{col_letter}{r}" for r in summary_rows_for_type] if summary_rows_for_type else ["0"]
            cell = ws.cell(row=current_row, column=g_cols[g], value=f"=SUM({','.join(sum_parts)})")
            cell.alignment = Alignment(horizontal="center", vertical="center"); cell.font = Font(bold=True); cell.border = Border(top=thick, bottom=thick)
            
        sum_parts_gsum = [f"{gsum_col_letter}{r}" for r in summary_rows_for_type] if summary_rows_for_type else ["0"]
        gsum_total_cell = ws.cell(row=current_row, column=g_sum_col, value=f"=SUM({','.join(sum_parts_gsum)})")
        gsum_total_cell.alignment = Alignment(horizontal="center", vertical="center"); gsum_total_cell.font = Font(bold=True); gsum_total_cell.border = Border(top=thick, bottom=thick)

    for gcol in vertical_thick_cols:
        col_idx = grid_start_col + gcol - 1
        for r in range(data_start_row, last_used_row + 1): 
            cell = ws.cell(row=r, column=col_idx)
            cell.border = Border(left=thick, right=cell.border.right, top=cell.border.top, bottom=cell.border.bottom)

    try:
        try: rs_total = int(rs_count_str or 0)
        except ValueError: rs_total = 0
        try: rpr = int(rd_per_rs_str or 0)
        except ValueError: rpr = 0
            
        sheet_name_for_formula = sheet_name.replace("'", "''")
        data_range_str = (
            f"'{sheet_name_for_formula}'!"
            f"{get_column_letter(grid_start_col)}{data_start_row}:"
            f"{get_column_letter(grid_right_col)}{last_used_row}"
        )
        summary_ws = wb.create_sheet(title="Summary")
        _write_summary_sheet_v5(summary_ws, rs_total, rpr, data_range_str)
    except Exception as e:
        print(f"Error creating summary sheet: {e}")

    wb.save(filename)

def _write_summary_sheet_v5(ws, rs_total: int, rpr: int, data_range_str: str):
    ws.font = Font(size=10)
    bold_font = Font(size=10, bold=True)
    stbd_font = Font(size=10, bold=True, color="974706")
    port_font = Font(size=10, bold=True, color="7030A0")
    title_font = Font(size=13, bold=True)
    header_fill = PatternFill("solid", fgColor="DDEEFF")
    yellow_fill = PatternFill("solid", fgColor="FFFF00")
    total_fill = PatternFill("solid", fgColor="F0F0F0")
    align_left_wrap = Alignment(horizontal="left", vertical="center", wrap_text=True) 
    align_right = Alignment(horizontal="right", vertical="center")
    align_center_wrap = Alignment(horizontal="center", vertical="center", wrap_text=True) 
    table_border = Border(left=Side(style="thin"), right=Side(style="thin"), top=Side(style="thin"), bottom=Side(style="thin"))
    small_font = Font(size=9)
    lgsp_fill = PatternFill("solid", fgColor="EBF1DE")

    headers = ["RS", "RD", "REF. CON", "A", "BREAKER", "CABLE", "KVA"]
    col_widths = [5.5, 6, 8, 7.5, 8, 12, 7.5] 

    def apply_styles(cell, font=None, fill=None, align=None, border=None, num_format=None):
        if font: cell.font = font
        if fill: cell.fill = fill
        if align: cell.alignment = align
        if border: cell.border = border
        if num_format: cell.number_format = num_format

    def set_column_widths(c_offset: int, widths: list):
        for i, width in enumerate(widths, 0):
            col_letter = get_column_letter(c_offset + i)
            ws.column_dimensions[col_letter].width = round(width + 0.00001)
            ws.column_dimensions[col_letter].auto_size = False
    
    def process_rs_list(rs_list, s_row, c_offset, side):
        fan_data_start_row, fan_data_end_row = 4, 26
        fan_rated_a_range   = f"$Y${fan_data_start_row}:$Y${fan_data_end_row}"
        fan_port_qty_range  = f"$U${fan_data_start_row}:$U${fan_data_end_row}"
        fan_stbd_qty_range  = f"$R${fan_data_start_row}:$R${fan_data_end_row}"
        fan_port_lgsp_range = f"$V${fan_data_start_row}:$V${fan_data_end_row}"
        fan_stbd_lgsp_range = f"$S${fan_data_start_row}:$S${fan_data_end_row}"
        fan_eff_range       = f"$Z${fan_data_start_row}:$Z${fan_data_end_row}"
        fan_stbd_power_range = f"$AD${fan_data_start_row}:$AD${fan_data_end_row}"
        fan_port_power_range = f"$AE${fan_data_start_row}:$AE${fan_data_end_row}"

        cur_row = s_row
        rs_totals = {}

        if rpr <= 0 or not rs_list:
            for i, h in enumerate(headers):
                apply_styles(ws.cell(cur_row, c_offset+i, value=h), font=bold_font, fill=header_fill, align=align_center_wrap, border=table_border)
            cur_row += 1
            for i in range(len(headers)):
                apply_styles(ws.cell(cur_row, c_offset+i), border=table_border)
            ws.cell(cur_row, c_offset, "-")
            cur_row += 1

            lgsp_start_row, lgsp_end_row = cur_row, cur_row + 2
            ws.merge_cells(start_row=lgsp_start_row, start_column=c_offset, end_row=lgsp_end_row, end_column=c_offset)
            apply_styles(ws.cell(lgsp_start_row, c_offset, value="LGSP"), align=align_center_wrap, border=table_border, fill=lgsp_fill)
            
            lgsp_kva_cells_for_rs = []
            for r in range(lgsp_start_row, lgsp_end_row + 1):
                rd_cell = ws.cell(r, c_offset + 1)
                apply_styles(rd_cell, border=table_border, align=align_left_wrap, fill=lgsp_fill)
                rd_addr = rd_cell.coordinate

                qty_rng = fan_port_qty_range if side == "PORT" else fan_stbd_qty_range
                lgsp_rng = fan_port_lgsp_range if side == "PORT" else fan_stbd_lgsp_range
                a_formula = f"=IFERROR(SUMPRODUCT(--({lgsp_rng}={rd_addr}), {qty_rng}, {fan_rated_a_range}), 0)"
                
                a_cell = ws.cell(r, c_offset + 3, value=a_formula)
                apply_styles(a_cell, border=table_border, align=align_right, num_format='0.00', fill=lgsp_fill)
                a_addr = a_cell.coordinate

                brk_original = (
                    f'IF(ROUND({a_addr},2)<150,"160/150",IF(ROUND({a_addr},2)<160,"160/160",IF(ROUND({a_addr},2)<175,"250/175",'
                    f'IF(ROUND({a_addr},2)<200,"250/200",IF(ROUND({a_addr},2)<225,"250/225",IF(ROUND({a_addr},2)<250,"250/250",'
                    f'IF(ROUND({a_addr},2)<300,"400/300",IF(ROUND({a_addr},2)<350,"400/350",IF(ROUND({a_addr},2)<400,"400/400",'
                    f'IF(ROUND({a_addr},2)<500,"630/500","630/630"))))))))))'
                )
                brk = f'=IF({rd_addr}="", "-", {brk_original})'
                apply_styles(ws.cell(r, c_offset + 4, value=brk), border=table_border, align=align_center_wrap, fill=lgsp_fill)

                cable_original = (
                    f'IF({a_addr}<89,"HT25(89A)",IF({a_addr}<110,"HT35(110A)",IF({a_addr}<137,"HT50(137A)",'
                    f'IF({a_addr}<169,"HT70(169A)",IF({a_addr}<205,"HT95(205A)",IF({a_addr}<237,"HT95(237A)",'
                    f'IF({a_addr}<274,"2xHT50(137A)",IF({a_addr}<338,"2xT70(169A)",IF({a_addr}<410,"2xT95(237A)",'
                    f'IF({a_addr}<474,"2xT120(237A)",""))))))))))'
                )
                cable = f'=IF({rd_addr}="", "-", {cable_original})'
                apply_styles(ws.cell(r, c_offset + 5, value=cable), border=table_border, align=align_center_wrap, fill=lgsp_fill)

                pwr_rng = fan_port_power_range if side == "PORT" else fan_stbd_power_range
                kva_formula = f'=IFERROR(SUMPRODUCT(--({lgsp_rng}={rd_addr}), {pwr_rng}, 1/{fan_eff_range}), 0)'
                kva_cell = ws.cell(r, c_offset + 6, value=kva_formula)
                apply_styles(kva_cell, border=table_border, align=align_right, num_format='0.00', fill=lgsp_fill)
                lgsp_kva_cells_for_rs.append(kva_cell.coordinate)
                
                apply_styles(ws.cell(r, c_offset + 2), border=table_border, fill=lgsp_fill)

            cur_row = lgsp_end_row + 1
            apply_styles(ws.cell(cur_row, c_offset, value="Total"), font=bold_font, align=align_left_wrap, fill=total_fill, border=table_border)
            for i in range(1, len(headers)): apply_styles(ws.cell(cur_row, c_offset + i), fill=total_fill, border=table_border)
            cur_row += 2
            return cur_row, rs_totals

        for rs in rs_list:
            for i, h in enumerate(headers):
                apply_styles(ws.cell(cur_row, c_offset+i, value=h), font=bold_font, fill=header_fill, align=align_center_wrap, border=table_border)
            cur_row += 1
            rd_list = _rd_list_for_rs(rs, rpr)
            rs_start_row = cur_row
            
            if not rd_list:
                ws.merge_cells(start_row=cur_row, start_column=c_offset+1, end_row=cur_row, end_column=c_offset+6)
                apply_styles(ws.cell(cur_row, c_offset), font=bold_font, align=align_left_wrap, border=table_border)
                apply_styles(ws.cell(cur_row, c_offset+1), align=align_left_wrap, border=table_border)
                for i in range(2, 7): apply_styles(ws.cell(cur_row, c_offset+i), border=table_border)
                cur_row += 1
                rs_start_row_for_sum = -1
            else:
                for rd in rd_list:
                    ref_addr = f"{get_column_letter(c_offset + 2)}{cur_row}"
                    a_addr   = f"{get_column_letter(c_offset + 3)}{cur_row}"
                    apply_styles(ws.cell(cur_row, c_offset), border=table_border)
                    apply_styles(ws.cell(cur_row, c_offset+1, value=f"RD-{rd}"), align=align_left_wrap, border=table_border)
                    apply_styles(ws.cell(cur_row, c_offset+2, value=f"=COUNTIF({data_range_str},{rd})"), align=align_right, border=table_border, num_format='0')
                    apply_styles(ws.cell(cur_row, c_offset+3, value=f"={ref_addr}*17.4*0.9"), align=align_right, border=table_border, num_format='0.00')
                    
                    rs_brk = f'=IF(ROUND({a_addr},2)<500,"630/500",IF(ROUND({a_addr},2)<630,"630/630",IF(ROUND({a_addr},2)>=630,"800/800","800/700")))'
                    apply_styles(ws.cell(cur_row, c_offset+4, value=rs_brk), align=align_center_wrap, border=table_border)
                    rs_cable = f'=IF(ROUND({a_addr},2)<507,"3xT70(169A)",IF(ROUND({a_addr},2)<676,"4xT70(169A)",IF(ROUND({a_addr},2)<845,"5xT70(169A)","6xT70(169A)")))'
                    apply_styles(ws.cell(cur_row, c_offset+5, value=rs_cable), align=align_center_wrap, border=table_border)
                    apply_styles(ws.cell(cur_row, c_offset+6, value=f"={ref_addr}*10.6"), align=align_right, border=table_border, num_format='0.00')
                    cur_row += 1
                rs_end_row = cur_row - 1
                ws.merge_cells(start_row=rs_start_row, start_column=c_offset, end_row=rs_end_row, end_column=c_offset)
                apply_styles(ws.cell(rs_start_row, c_offset, value=f"RS-{rs}"), font=bold_font, align=align_center_wrap, border=table_border)
                for r in range(rs_start_row+1, rs_end_row+1): apply_styles(ws.cell(r, c_offset), border=table_border)
                rs_start_row_for_sum = rs_start_row

            lgsp_kva_cells_for_rs = []
            lgsp_start_row, lgsp_end_row = cur_row, cur_row + 2
            ws.merge_cells(start_row=lgsp_start_row, start_column=c_offset, end_row=lgsp_end_row, end_column=c_offset)
            apply_styles(ws.cell(lgsp_start_row, c_offset, value="LGSP"), align=align_center_wrap, border=table_border, fill=lgsp_fill)

            for r in range(lgsp_start_row, lgsp_end_row + 1):
                rd_cell = ws.cell(r, c_offset + 1)
                apply_styles(rd_cell, border=table_border, align=align_left_wrap, fill=lgsp_fill)
                rd_addr = rd_cell.coordinate
                
                qty_rng = fan_port_qty_range if side == "PORT" else fan_stbd_qty_range
                lgsp_rng = fan_port_lgsp_range if side == "PORT" else fan_stbd_lgsp_range
                pwr_rng = fan_port_power_range if side == "PORT" else fan_stbd_power_range

                a_formula = f"=IFERROR(SUMPRODUCT(--({lgsp_rng}={rd_addr}), {qty_rng}, {fan_rated_a_range}), 0)"
                a_cell = ws.cell(r, c_offset + 3, value=a_formula)
                apply_styles(a_cell, border=table_border, align=align_right, num_format='0.00', fill=lgsp_fill)
                a_addr = a_cell.coordinate

                brk_original = (
                    f'IF(ROUND({a_addr},2)<150,"160/150",IF(ROUND({a_addr},2)<160,"160/160",IF(ROUND({a_addr},2)<175,"250/175",'
                    f'IF(ROUND({a_addr},2)<200,"250/200",IF(ROUND({a_addr},2)<225,"250/225",IF(ROUND({a_addr},2)<250,"250/250",'
                    f'IF(ROUND({a_addr},2)<300,"400/300",IF(ROUND({a_addr},2)<350,"400/350",IF(ROUND({a_addr},2)<400,"400/400",'
                    f'IF(ROUND({a_addr},2)<500,"630/500","630/630"))))))))))'
                )
                brk = f'=IF({rd_addr}="", "-", {brk_original})'
                apply_styles(ws.cell(r, c_offset + 4, value=brk), border=table_border, align=align_center_wrap, fill=lgsp_fill)

                cable_original = (
                    f'IF({a_addr}<89,"HT25(89A)",IF({a_addr}<110,"HT35(110A)",IF({a_addr}<137,"HT50(137A)",'
                    f'IF({a_addr}<169,"HT70(169A)",IF({a_addr}<205,"HT95(205A)",IF({a_addr}<237,"HT95(237A)",'
                    f'IF({a_addr}<274,"2xHT50(137A)",IF({a_addr}<338,"2xT70(169A)",IF({a_addr}<410,"2xT95(237A)",'
                    f'IF({a_addr}<474,"2xT120(237A)",""))))))))))'
                )
                cable = f'=IF({rd_addr}="", "-", {cable_original})'
                apply_styles(ws.cell(r, c_offset + 5, value=cable), border=table_border, align=align_center_wrap, fill=lgsp_fill)

                kva_formula = f'=IFERROR(SUMPRODUCT(--({lgsp_rng}={rd_addr}), {pwr_rng}, 1/{fan_eff_range}), 0)'
                kva_cell = ws.cell(r, c_offset + 6, value=kva_formula)
                apply_styles(kva_cell, border=table_border, align=align_right, num_format='0.00', fill=lgsp_fill)
                lgsp_kva_cells_for_rs.append(kva_cell.coordinate)
                apply_styles(ws.cell(r, c_offset + 2), border=table_border, fill=lgsp_fill)

            cur_row = lgsp_end_row + 1
            apply_styles(ws.cell(cur_row, c_offset, value="Total"), font=bold_font, align=align_left_wrap, fill=total_fill, border=table_border)
            
            if rs_start_row_for_sum == -1:
                ref_sum, a_sum, kva_sum = 0, 0, 0
                ref_total_coord = "0"
            else:
                rs_end_row_for_sum = lgsp_start_row - 1
                ref_sum = f"=SUM({get_column_letter(c_offset+2)}{rs_start_row_for_sum}:{get_column_letter(c_offset+2)}{rs_end_row_for_sum})"
                a_sum   = f"=SUM({get_column_letter(c_offset+3)}{rs_start_row_for_sum}:{get_column_letter(c_offset+3)}{rs_end_row_for_sum})"
                kva_sum = f"=SUM({get_column_letter(c_offset+6)}{rs_start_row_for_sum}:{get_column_letter(c_offset+6)}{rs_end_row_for_sum})"
                ref_total_coord = f"{get_column_letter(c_offset+2)}{cur_row}"

            apply_styles(ws.cell(cur_row, c_offset + 1), fill=total_fill, border=table_border)
            rs_totals[rs] = {'ref_total': ref_total_coord, 'lgsp_kva_cells': lgsp_kva_cells_for_rs}
            apply_styles(ws.cell(cur_row, c_offset + 2, value=ref_sum), font=bold_font, align=align_right, fill=total_fill, border=table_border, num_format='0')
            apply_styles(ws.cell(cur_row, c_offset + 3, value=a_sum),   font=bold_font, align=align_right, fill=total_fill, border=table_border, num_format='0.00')
            apply_styles(ws.cell(cur_row, c_offset + 4), fill=total_fill, border=table_border)
            apply_styles(ws.cell(cur_row, c_offset + 5), fill=total_fill, border=table_border)
            apply_styles(ws.cell(cur_row, c_offset + 6, value=kva_sum), font=bold_font, align=align_right, fill=total_fill, border=table_border, num_format='0.00')
            cur_row += 2
        return cur_row, rs_totals

    port_col_offset = 1
    even_rs = [i for i in range(1, rs_total + 1) if i % 2 == 0]
    stbd_col_offset = port_col_offset + len(col_widths) + 1
    odd_rs  = [i for i in range(1, rs_total+1) if i % 2 == 1]

    set_column_widths(port_col_offset, col_widths)
    set_column_widths(stbd_col_offset, col_widths)
    ws.column_dimensions[get_column_letter(port_col_offset + len(col_widths))].width = 5 

    ws.merge_cells(start_row=2, start_column=port_col_offset, end_row=2, end_column=port_col_offset + len(headers) - 1)
    apply_styles(ws.cell(2, port_col_offset, "PORT"), font=title_font, align=align_center_wrap)
    ws.merge_cells(start_row=2, start_column=stbd_col_offset, end_row=2, end_column=stbd_col_offset + len(headers) - 1)
    apply_styles(ws.cell(2, stbd_col_offset, "STBD"), font=title_font, align=align_center_wrap)
    ws.row_dimensions[2].height = 25
    
    port_next, port_totals = process_rs_list(even_rs, 3, port_col_offset, "PORT")
    stbd_next, stbd_totals = process_rs_list(odd_rs, 3, stbd_col_offset, "STBD")
    all_rs_totals = {**port_totals, **stbd_totals}

    fan_table_start_col = stbd_col_offset + len(headers) + 1
    ws.column_dimensions[get_column_letter(fan_table_start_col - 1)].width = 10
    
    table1_data_widths = [23, 8.5, 9, 23, 8.5, 9, 10.5, 10.5, 10.5, 8, 6, 6, 10.5, 10.5, 10.5]
    set_column_widths(fan_table_start_col, table1_data_widths)

    ws.merge_cells(start_row=1, start_column=fan_table_start_col, end_row=1, end_column=fan_table_start_col + len(table1_data_widths) - 1)
    apply_styles(ws.cell(1, fan_table_start_col, "Reefer section에서 배전되는 Fans"), font=bold_font, align=align_left_wrap)

    q_col = fan_table_start_col
    ws.merge_cells(start_row=2, start_column=q_col, end_row=2, end_column=q_col+1)
    apply_styles(ws.cell(2, q_col, "Cargo hold Fan (STBD)"), font=stbd_font, fill=yellow_fill, align=align_center_wrap, border=table_border)
    apply_styles(ws.cell(3, q_col, "Description"), font=stbd_font, fill=yellow_fill, align=align_center_wrap, border=table_border)
    apply_styles(ws.cell(3, q_col+1, "Q'ty"), font=stbd_font, fill=yellow_fill, align=align_center_wrap, border=table_border)
    
    s_col = q_col + 2
    ws.merge_cells(start_row=2, start_column=s_col, end_row=3, end_column=s_col)
    apply_styles(ws.cell(2, s_col, "LGSP No."), font=bold_font, fill=yellow_fill, align=align_center_wrap, border=table_border)

    t_col = s_col + 1
    ws.merge_cells(start_row=2, start_column=t_col, end_row=2, end_column=t_col+1)
    apply_styles(ws.cell(2, t_col, "Cargo hold Fan (PORT)"), font=port_font, fill=yellow_fill, align=align_center_wrap, border=table_border)
    apply_styles(ws.cell(3, t_col, "Description"), font=port_font, fill=yellow_fill, align=align_center_wrap, border=table_border)
    apply_styles(ws.cell(3, t_col+1, "Q'ty"), font=port_font, fill=yellow_fill, align=align_center_wrap, border=table_border)

    v_col = t_col + 2
    ws.merge_cells(start_row=2, start_column=v_col, end_row=3, end_column=v_col)
    apply_styles(ws.cell(2, v_col, "LGSP No."), font=bold_font, fill=yellow_fill, align=align_center_wrap, border=table_border)

    merged_headers = ["Total No. of C/Hold fans", "C/Hold fan capacity (Rated, kW)", "C/Hold fan capacity (Rated, A)", 
                      "Efficiency", "Load factor", "Div. factor", "Actual power consumption (kW)", 
                      "Actual power consumption(STBD, kW)", "Actual power consumption(PORT, kW)"]
    w_col = v_col + 1
    y_h = {"TR capacity (SPEC)", "C/Hold fan capacity (Rated, kW)", "C/Hold fan capacity (Rated, A)", "Efficiency"}
    for i, h in enumerate(merged_headers):
        ci = w_col + i
        ws.merge_cells(start_row=2, start_column=ci, end_row=3, end_column=ci)
        fnt = stbd_font if "STBD" in h else (port_font if "PORT" in h else bold_font)
        fil = yellow_fill if h in y_h else header_fill
        apply_styles(ws.cell(2, ci, h), font=fnt, fill=fil, align=align_center_wrap, border=table_border)

    titles = ["SIDE PASSGAEWAY(FWD)", "No.1A (Exp)", "No.2F (Exp)", "No.3F", "No.4F", "SIDE PASSAGEWAY(AFT)",
              "No.5F1", "No.5F2", "No.5A1", "No.5A2", "No.6F1", "No.6F2", "No.6A1", "No.6A2", "No.7F", "No.8F",
              "PIPE DUCT", "No.9F"]
    cur_row = 4
    for t in titles:
        apply_styles(ws.cell(cur_row, q_col, t), align=align_left_wrap, border=table_border)
        apply_styles(ws.cell(cur_row, t_col, t), align=align_left_wrap, border=table_border)
        
        sq = get_column_letter(q_col+1)
        pq = get_column_letter(t_col+1)
        kw = get_column_letter(w_col+1)
        eff = get_column_letter(w_col+3)
        ld = get_column_letter(w_col+4)
        div = get_column_letter(w_col+5)
        act = get_column_letter(w_col+6)
        tot = get_column_letter(w_col)

        for i in range(len(table1_data_widths)):
            ci = fan_table_start_col + i
            cell = ws.cell(cur_row, ci)
            if ci not in [q_col, t_col]: apply_styles(cell, border=table_border)
            if ci == w_col: 
                cell.value = f"=SUM(${sq}{cur_row},${pq}{cur_row})"
                cell.number_format = '0'
            elif ci == w_col + 3: cell.value = 0.88
            elif ci == w_col + 4: cell.value = 0.8
            elif ci == w_col + 5: cell.value = 1
            elif ci == w_col + 6: cell.value = f"=IFERROR((${kw}{cur_row}/${eff}{cur_row})*${ld}{cur_row}*${div}{cur_row}, 0)"
            elif ci == w_col + 7: cell.value = f"=IFERROR((${act}{cur_row}/${tot}{cur_row})*${sq}{cur_row}, 0)"
            elif ci == w_col + 8: cell.value = f"=IFERROR((${act}{cur_row}/${tot}{cur_row})*${pq}{cur_row}, 0)"
            
        cur_row += 1
    
    for _ in range(5):
        apply_styles(ws.cell(cur_row, q_col), border=table_border)
        apply_styles(ws.cell(cur_row, t_col), border=table_border)
        for i in range(len(table1_data_widths)):
            if fan_table_start_col+i not in [q_col, t_col]:
                apply_styles(ws.cell(cur_row, fan_table_start_col+i), border=table_border)
        cur_row += 1

    cur_row += 2
    reefer_start_row = cur_row
    h2 = ["Reefer Section board", "Ref. CNTR (FEU)", "Ref. CNTR (kVA)", "Cargo fan(kVA)", "Total Cap (kVA)", "TR Cap (kVA)", "TR Cap (SPEC)"]
    ws.merge_cells(start_row=cur_row-1, start_column=fan_table_start_col, end_row=cur_row-1, end_column=fan_table_start_col+len(h2)-1)
    apply_styles(ws.cell(cur_row-1, fan_table_start_col, "Reefer CNTR 계산"), font=bold_font, align=align_left_wrap)

    for i, h in enumerate(h2):
        ci = fan_table_start_col + i
        ws.merge_cells(start_row=cur_row, start_column=ci, end_row=cur_row+1, end_column=ci)
        fil = yellow_fill if "SPEC" in h else header_fill
        apply_styles(ws.cell(cur_row, ci, h), font=bold_font, fill=fil, align=align_center_wrap, border=table_border)
        apply_styles(ws.cell(cur_row+1, ci), fill=fil, border=table_border)
    
    cur_row += 2
    colors = ["EBF1DE", "D8E4BC", "E4DFEC", "CCC0DA", "DAEEF3", "B7DEE8"]
    
    tr_k_addr = f"${get_column_letter(fan_table_start_col+1)}${reefer_start_row + rs_total + 4}" # Approximation
    tr_eff_addr = f"${get_column_letter(fan_table_start_col+1)}${reefer_start_row + rs_total + 5}"

    vertical_headers = []
    if rs_total > 0:
        for i in range(1, rs_total + 1):
            suffix = "(S)" if i % 2 == 1 else "(P)"
            vertical_headers.append(f"RS-{i} {suffix}")

    for idx, title in enumerate(vertical_headers):
        rs_idx = idx + 1
        clr = colors[idx % len(colors)]
        fil = PatternFill("solid", fgColor=clr)
        apply_styles(ws.cell(cur_row, fan_table_start_col, title), font=bold_font, fill=fil, align=align_left_wrap, border=table_border)
        
        rs_d = all_rs_totals.get(rs_idx, {})
        ref_tot = rs_d.get('ref_total', "0")
        lgsp_list = rs_d.get('lgsp_kva_cells', [])
        
        ref_cell = ws.cell(cur_row, fan_table_start_col+1, f"={ref_tot}")
        apply_styles(ref_cell, fill=fil, border=table_border, align=align_right, num_format='0')
        
        apply_styles(ws.cell(cur_row, fan_table_start_col+2), fill=fil, border=table_border, align=align_right, num_format='0.00') # 수식은 나중에
        
        cf_form = f"=SUM({','.join(lgsp_list)})" if lgsp_list else "0"
        cf_cell = ws.cell(cur_row, fan_table_start_col+3, cf_form)
        apply_styles(cf_cell, fill=fil, border=table_border, align=align_right, num_format='0.00')

        tot_form = f"=SUM({get_column_letter(fan_table_start_col+2)}{cur_row},{cf_cell.coordinate})"
        apply_styles(ws.cell(cur_row, fan_table_start_col+4, tot_form), fill=fil, border=table_border, align=align_right, num_format='0.00')
        
        apply_styles(ws.cell(cur_row, fan_table_start_col+5), fill=fil, border=table_border, align=align_right, num_format='0.00')
        apply_styles(ws.cell(cur_row, fan_table_start_col+6), fill=fil, border=table_border)
        cur_row += 1

    tr_start = cur_row + 2
    apply_styles(ws.cell(tr_start, fan_table_start_col, "Reefer cntr for TR"), font=bold_font, align=align_left_wrap, border=table_border)
    tr_k_cell = ws.cell(tr_start, fan_table_start_col+1, 10.6)
    apply_styles(tr_k_cell, align=align_right, border=table_border, num_format='0.0" kVA"')
    tr_k_addr = f"${get_column_letter(tr_k_cell.column)}${tr_k_cell.row}"

    apply_styles(ws.cell(tr_start+1, fan_table_start_col, "TR efficiency"), font=bold_font, align=align_left_wrap, border=table_border)
    tr_eff_cell = ws.cell(tr_start+1, fan_table_start_col+1, 0.95)
    apply_styles(tr_eff_cell, align=align_right, border=table_border, num_format='0.00')
    tr_eff_addr = f"${get_column_letter(tr_eff_cell.column)}${tr_eff_cell.row}"

    ws.cell(tr_start+1, fan_table_start_col+2, "실제 TR도면 접수 후 다시 기입").font = small_font

    # Fill formulas
    for r in range(reefer_start_row + 2, cur_row):
        ref_cnt_coord = ws.cell(r, fan_table_start_col+1).coordinate
        ws.cell(r, fan_table_start_col+2).value = f"={ref_cnt_coord}*{tr_k_addr}"
        
        tot_cap_coord = ws.cell(r, fan_table_start_col+4).coordinate
        ws.cell(r, fan_table_start_col+5).value = f"=IFERROR({tot_cap_coord}/{tr_eff_addr}, 0)"


class GridCanvas(ttk.Frame):
    def __init__(self, master):
        super().__init__(master, padding=10)
        self.pack(fill="both", expand=True)

        self.rows, self.cols, self.cell_px = 4, 20, 22
        self.cell_colors, self.cell_numbers, self.sockets = {}, {}, set()
        
        self.mode_color = tk.BooleanVar(value=False) 
        self.mode_number = tk.BooleanVar(value=False)
        self.mode_eraser = tk.BooleanVar(value=False)
        self.mode_socket = tk.BooleanVar(value=False)

        self.current_label = tk.StringVar(value="1")
        self.FIXED_COLORS = ["#FFFF99", "#99CCFF", "#CCFFCC", "#F2DCDB"] 
        self.current_color = tk.StringVar(value=self.FIXED_COLORS[0])
        
        self.ship_no = tk.StringVar(value="")
        self.looking_txt = tk.StringVar(value="LOOKING TO FWD") 
        self.rs_count = tk.StringVar(value="0")
        self.rd_per_rs = tk.StringVar(value="0")
        
        self.gang_vars = {g: tk.StringVar(value="0") for g in range(3, 10)}
        self.items: List[Item] = []
        self.editing_index: Optional[int] = None
        self.drag_start, self.drag_rect_id = None, None
        
        self.drag_start_right, self.drag_rect_id_right = None, None
        self.insert_pos = tk.StringVar(value="1")
        self.undo_stack = []

        # [PDF Export & Project Save State]
        self.last_imported_pdf_path: Optional[str] = None
        self.last_grouped_tokens: Optional[pd.DataFrame] = None
        self.group_to_block_map: Dict[int, int] = {}

        self.rs_count.trace_add("write", self._recompute_all)
        self.rd_per_rs.trace_add("write", self._recompute_all)

        self._create_widgets()
        self.draw_all()
        self._recompute_all()
        self._update_insert_spin_range()
        self._update_ui_visibility()

    def set_current_color(self, hx: str):
        self.current_color.set(hx)
        self.color_preview.config(bg=hx)

    def pick_color_dialog(self):
        _, hx = colorchooser.askcolor(color=self.current_color.get(), title="Pick Color")
        if hx:
            self.set_current_color(hx)

    def _toggle_mode(self, mode_var: tk.BooleanVar, exclusive: bool, button: ttk.Button):
        if exclusive:
            was_on = mode_var.get()
            self.mode_color.set(False)
            self.mode_number.set(False)
            self.mode_eraser.set(False)
            self.mode_socket.set(False)
            mode_var.set(was_on if False else (not was_on))
        else:
            if self.mode_eraser.get() or self.mode_socket.get():
                self.mode_eraser.set(False)
                self.mode_socket.set(False)
                mode_var.set(True)
            else:
                mode_var.set(not mode_var.get())
        self._update_all_button_appearances()
        self._update_ui_visibility()

    def _update_button_appearance(self, button: ttk.Button, is_selected: bool):
        if is_selected:
            button.state(['pressed']) 
        else:
            button.state(['!pressed'])

    def _update_all_button_appearances(self):
        self._update_button_appearance(self.btn_color, self.mode_color.get())
        self._update_button_appearance(self.btn_number, self.mode_number.get())
        self._update_button_appearance(self.btn_eraser, self.mode_eraser.get())
        self._update_button_appearance(self.btn_socket, self.mode_socket.get())
        
    def _update_ui_visibility(self):
        if self.mode_number.get():
            self.number_frame.grid(row=1, column=2, columnspan=2, sticky="ew")
        else:
            self.number_frame.grid_forget()
        if self.mode_color.get():
            self.color_frame.grid(row=1, column=4, columnspan=5, sticky="ew")
        else:
            self.color_frame.grid_forget()
            
    def _create_widgets(self):
        # 1. 메인 윈도우(Root) 가져오기
        root = self.winfo_toplevel()
        
        # 2. 메뉴바 생성
        menubar = tk.Menu(root)
        
        # 3. [File] 메뉴 생성 및 항목 추가
        file_menu = tk.Menu(menubar, tearoff=0)
        file_menu.add_command(label="Import PDF...", command=self.import_from_pdf)
        file_menu.add_command(label="Export Annotated PDF...", command=self.export_annotated_pdf)
        file_menu.add_separator() # 구분선
        file_menu.add_command(label="Save Project...", command=self.save_project)
        file_menu.add_command(label="Load Project...", command=self.load_project)
        file_menu.add_separator()
        file_menu.add_command(label="Export Excel...", command=self.export_excel) # 엑셀 내보내기도 메뉴에 넣으면 깔끔합니다 (선택사항)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=root.quit)
        
        # 4. 메뉴바에 [File] 메뉴 등록
        menubar.add_cascade(label="File", menu=file_menu)
        
        # 5. 루트 윈도우에 메뉴바 적용
        root.config(menu=menubar)

        # --- 기존 레이아웃 코드 시작 ---
        outer = ttk.Frame(self); outer.pack(fill="both", expand=True)
        right = ttk.Frame(outer); right.pack(side="right", fill="y", padx=(10, 0))
        left = ttk.Frame(outer); left.pack(side="left", fill="both", expand=True)

        cfg = ttk.LabelFrame(right, text="RS / RD Panels", padding=10)
        cfg.pack(fill="x", pady=(0, 10))
        ttk.Label(cfg, text="RS Panels").grid(row=0, column=0, sticky="e", padx=5, pady=2)
        ttk.Spinbox(cfg, from_=0, to=999, textvariable=self.rs_count, width=8).grid(row=0, column=1, sticky="w", pady=2)
        ttk.Label(cfg, text="RD per RS").grid(row=0, column=2, sticky="e", padx=5, pady=2)
        ttk.Spinbox(cfg, from_=0, to=999, textvariable=self.rd_per_rs, width=8).grid(row=0, column=3, sticky="w", pady=2)

        summary_frame = ttk.LabelFrame(right, text="Live Summary", padding=10)
        summary_frame.pack(fill="both", expand=True, pady=(0, 10)) 

        self.grand_total_label = ttk.Label(summary_frame, text="Grand Total: 0", font=("Segoe UI", 9, "bold"))
        self.grand_total_label.pack(anchor="w", pady=(0, 5))

        alloc = ttk.Frame(summary_frame); alloc.pack(fill="both", expand=True)
        left_col = ttk.Frame(alloc); left_col.pack(side="left", fill="both", expand=True, padx=(0,5))
        ttk.Label(left_col, text="PORT RS").pack(anchor="w")
        self.txt_even = tk.Text(left_col, width=15, height=34, font=("Consolas", 10)); self.txt_even.pack(fill="both", expand=True); self.txt_even.configure(state="disabled")

        right_col = ttk.Frame(alloc); right_col.pack(side="left", fill="both", expand=True, padx=(5,0))
        ttk.Label(right_col, text="STBD RS").pack(anchor="w")
        self.txt_odd = tk.Text(right_col, width=15, height=34, font=("Consolas", 10)); self.txt_odd.pack(fill="both", expand=True); self.txt_odd.configure(state="disabled")

        top = ttk.Frame(left); top.pack(fill="x", pady=(0, 10))
        top.columnconfigure(6, weight=1) 
        
        ttk.Label(top, text="Ship No.").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(top, textvariable=self.ship_no, width=14).grid(row=0, column=1, sticky="w", padx=(0,12), pady=2)
        
        mode_btn_frame = ttk.Frame(top)
        mode_btn_frame.grid(row=0, column=2, columnspan=8, sticky="w", padx=4, pady=2)
        
        self.btn_color = ttk.Button(mode_btn_frame, text="COLOR", command=lambda: self._toggle_mode(self.mode_color, False, self.btn_color))
        self.btn_color.pack(side="left", padx=2)
        self.btn_number = ttk.Button(mode_btn_frame, text="NUMBER", command=lambda: self._toggle_mode(self.mode_number, False, self.btn_number))
        self.btn_number.pack(side="left", padx=2)
        self.btn_eraser = ttk.Button(mode_btn_frame, text="ERASER", command=lambda: self._toggle_mode(self.mode_eraser, True, self.btn_eraser))
        self.btn_eraser.pack(side="left", padx=(10, 2))
        self.btn_socket = ttk.Button(mode_btn_frame, text="SOCKET", command=lambda: self._toggle_mode(self.mode_socket, True, self.btn_socket))
        self.btn_socket.pack(side="left", padx=2)

        self._update_all_button_appearances()

        self.number_frame = ttk.Frame(top)
        ttk.Label(self.number_frame, text="Number").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Entry(self.number_frame, textvariable=self.current_label, width=10).grid(row=0, column=1, sticky="w", padx=(0,12), pady=2)
        
        self.color_frame = ttk.Frame(top)
        ttk.Label(self.color_frame, text="Color").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        color_col_start = 1
        for i, color in enumerate(self.FIXED_COLORS):
            btn = tk.Button(self.color_frame, text="   ", bg=color, width=2, relief="raised", command=lambda c=color: self.set_current_color(c))
            btn.grid(row=0, column=color_col_start + i, sticky="w", padx=(1, 1), pady=2)
        self.color_preview = tk.Label(self.color_frame, text="   ", bg=self.current_color.get(), relief="groove", width=4)
        self.color_preview.grid(row=0, column=color_col_start + len(self.FIXED_COLORS), sticky="w", padx=(0,4), pady=2)
        ttk.Button(self.color_frame, text="Pick…", command=self.pick_color_dialog).grid(row=0, column=color_col_start + len(self.FIXED_COLORS) + 1, sticky="w", padx=(2,12), pady=2)

        # [수정됨] Main Button Row: 메뉴로 이동한 버튼들을 제외하고 남은 버튼들만 배치
        btn_row = ttk.Frame(left); btn_row.pack(fill="x", pady=5)

        # 메뉴바에 없는 나머지 편집/설정 버튼들
        ttk.Button(btn_row, text="Save Block", command=self.save_edits).grid(row=0, column=0, padx=2, pady=2)
        ttk.Button(btn_row, text="Auto Set", command=self.auto_set_groups).grid(row=0, column=1, padx=2, pady=2)
        ttk.Button(btn_row, text="Undo", command=self.undo).grid(row=0, column=2, padx=2, pady=2)
        ttk.Button(btn_row, text="Clear All", command=self.clear_all).grid(row=0, column=3, padx=2, pady=2)
        
        # Export Excel은 메뉴에도 넣었지만, 자주 쓴다면 버튼으로도 남겨둘 수 있습니다. (여기선 버튼에서는 제거하고 메뉴로 통합했다고 가정)
        # 만약 버튼으로도 남기고 싶다면 아래 주석을 해제하세요.
        # ttk.Button(btn_row, text="Export Excel…", command=self.export_excel).grid(row=0, column=4, padx=(12,2), pady=2)
        
        self.canvas = tk.Canvas(left, bg="white", highlightthickness=1, highlightbackground="#cccccc")
        self.canvas.pack(fill="both", expand=True, pady=5)
        self.canvas.bind("<Button-1>", self.on_press); self.canvas.bind("<B1-Motion>", self.on_drag); self.canvas.bind("<ButtonRelease-1>", self.on_release)
        self.canvas.bind("<Button-3>", self.on_press_right); self.canvas.bind("<B3-Motion>", self.on_drag_right); self.canvas.bind("<ButtonRelease-3>", self.on_release_right)
        self.canvas.bind("<Control-Button-1>", self.on_press_right); self.canvas.bind("<Control-B1-Motion>", self.on_drag_right); self.canvas.bind("<Control-ButtonRelease-1>", self.on_release_right)

        gang_row = ttk.Frame(left); gang_row.pack(fill="x", pady=5)
        ttk.Label(gang_row, text="Gang per Block").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        col = 1

        for g in range(3, 10):
            ttk.Label(gang_row, text=f"{g}").grid(row=0, column=col, sticky="e", padx=4, pady=2)
            ttk.Entry(gang_row, textvariable=self.gang_vars[g], width=5).grid(row=0, column=col+1, sticky="w", padx=(0,8), pady=2)
            col += 2

        bottom = ttk.Frame(left); bottom.pack(fill="x", pady=5)
        ttk.Label(bottom, text="Items (Blocks / Sections)").pack(anchor="w")
        self.items_list = tk.Listbox(bottom, height=8); self.items_list.pack(fill="x", expand=True)
        self.items_list.bind("<Double-Button-1>", lambda e: self.edit_selected())
        self.items_list.bind("<Button-3>", self._show_context_menu)
        self.context_menu = tk.Menu(self.items_list, tearoff=0)
        self.context_menu.add_command(label="Delete Block", command=lambda: self.context_menu_action("DELETE"))
        self.context_menu.add_separator()
        self.context_menu.add_command(label="Insert ACCOMMODATION", command=lambda: self.context_menu_action("INSERT_ACCOMMODATION"))
        self.context_menu.add_command(label="Insert E/R CASING", command=lambda: self.context_menu_action("INSERT_E/R CASING"))
        
        meta = ttk.LabelFrame(left, text="Block Meta", padding=10)
        meta.pack(fill="x", pady=5)   
        self.meta_label_type = tk.StringVar(value="Hatch No")
        ttk.Label(meta, text="Label Type").grid(row=0, column=0, sticky="e", padx=4, pady=2)
        ttk.Combobox(meta, textvariable=self.meta_label_type, values=["Hatch No", "Hold No"], width=10, state="readonly").grid(row=0, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(meta, text="No.").grid(row=0, column=2, sticky="e", padx=8, pady=2)
        self.meta_no = tk.StringVar(value="")
        ttk.Entry(meta, textvariable=self.meta_no, width=10).grid(row=0, column=3, sticky="w", padx=4, pady=2)
        ttk.Label(meta, text="Bay").grid(row=1, column=0, sticky="e", padx=4, pady=2)
        self.meta_bay = tk.StringVar(value="")
        ttk.Entry(meta, textvariable=self.meta_bay, width=10).grid(row=1, column=1, sticky="w", padx=4, pady=2)
        ttk.Label(meta, text="Deck").grid(row=1, column=2, sticky="e", padx=8, pady=2)
        self.meta_deck = tk.StringVar(value="")
        ttk.Entry(meta, textvariable=self.meta_deck, width=12).grid(row=1, column=3, sticky="w", padx=4, pady=2)
        ttk.Button(meta, text="Apply to Selected", command=self.apply_meta_to_selected).grid(row=0, column=4, rowspan=2, sticky="nsw", padx=(12,4), pady=2)
        
    def _save_state_for_undo(self):
        state = {
            'items': copy.deepcopy(self.items),
            'editing_index': self.editing_index,
            'canvas_data': (copy.deepcopy(self.cell_colors), copy.deepcopy(self.cell_numbers), copy.deepcopy(self.sockets))
        }
        self.undo_stack.append(state)
        if len(self.undo_stack) > 10:
            self.undo_stack.pop(0)

    def undo(self):
        if not self.undo_stack:
            messagebox.showinfo("Undo", "더 이상 되돌릴 작업이 없습니다.")
            return
        state = self.undo_stack.pop()
        self.items = state['items']
        self.editing_index = state['editing_index']
        self.cell_colors, self.cell_numbers, self.sockets = state['canvas_data']
        self.mode_color.set(False); self.mode_number.set(False); self.mode_eraser.set(False); self.mode_socket.set(False)
        self._update_all_button_appearances()
        self._update_ui_visibility()
        if self.editing_index is not None:
            self.edit_selected()
        else:
            self.draw_all()
        self.refresh_list()
        self._recompute_all()

    def clear_all(self):
        if messagebox.askyesno("Clear All", "모든 블럭과 작업 내용을 정말로 초기화하시겠습니까?"):
            self._save_state_for_undo()
            self.items.clear()
            self.editing_index = None
            self.clear_canvas()
            self.refresh_list()
            self.winfo_toplevel().title("Reefer Layout")

    def _recompute_all(self, *args):
        self._update_allocation_display()

    def _update_allocation_display(self):
        try:
            rs_total = int(self.rs_count.get() or 0)
            rpr = int(self.rd_per_rs.get() or 0)
        except(ValueError, TypeError):
            rs_total = 0
            rpr = 0

        even_rs = [i for i in range(1, rs_total+1) if i % 2 == 0]
        odd_rs  = [i for i in range(1, rs_total+1) if i % 2 == 1]
        rd_counts = _collect_rd_counts(self.items)
        grand_total = sum(rd_counts.values())
        self.grand_total_label.config(text=f"Grand Total: {grand_total}")
        for txt_widget, indices in [(self.txt_even, even_rs), (self.txt_odd, odd_rs)]:
            content = _build_rs_summary(indices, rd_counts, rpr)
            txt_widget.config(state="normal")
            txt_widget.delete("1.0", tk.END)
            txt_widget.insert(tk.END, content)
            txt_widget.config(state="disabled")

    def clear_canvas(self):
        self.cell_colors.clear(); self.cell_numbers.clear(); self.sockets.clear()
        self.draw_all()

    def cell_to_xy(self, r, c):
        x1, y1 = (c - 1) * self.cell_px + 1, (r - 1) * self.cell_px + 1
        return x1, y1, x1 + self.cell_px, y1 + self.cell_px

    def xy_to_cell(self, x, y):
        c, r = int((x - 1) // self.cell_px) + 1, int((y - 1) // self.cell_px) + 1
        return (r, c) if 1 <= r <= self.rows and 1 <= c <= self.cols else None

    def draw_all(self):
        self.canvas.delete("all")
        self.canvas.config(width=self.cols * self.cell_px + 2, height=self.rows * self.cell_px + 2)
        for r in range(1, self.rows + 1):
            for c in range(1, self.cols + 1):
                x1, y1, x2, y2 = self.cell_to_xy(r, c)
                self.canvas.create_rectangle(x1, y1, x2, y2, fill=self.cell_colors.get((r, c), "#FFFFFF"), outline="#CCCCCC")
        for r, c in self.sockets:
            x1, y1, x2, y2 = self.cell_to_xy(r, c)
            self.canvas.create_rectangle(x1 + 1, y1 + 1, x2 - 1, y2 - 1, outline="#FF0000", width=2)
        for (r, c), val in self.cell_numbers.items():
            x, y = (c - 0.5) * self.cell_px, (r - 0.5) * self.cell_px
            self.canvas.create_text(x, y, text=str(val), font=("Arial", 8))
            
# [추가할 코드] GridCanvas 클래스 내부 메서드로 추가
    def cell_to_rect_global(self, idx, r, c):
        """특정 블록(idx)의 (r, c) 셀에 해당하는 캔버스 글로벌 좌표(x1, y1, x2, y2)를 반환"""
        for info in self.block_layout_info:
            # info 구조: (idx, y_start, y_end, x_start, x_end, rows, cols)
            if info[0] == idx:
                _, y_start, _, x_start, _, _, _ = info
                x1 = x_start + (c - 1) * self.cell_px
                y1 = y_start + (r - 1) * self.cell_px
                x2 = x1 + self.cell_px
                y2 = y1 + self.cell_px
                return x1, y1, x2, y2
        return None
        
    def on_press(self, e):
        self._save_state_for_undo()
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        res = self.xy_to_cell_global(cx, cy)
        
        if res:
            self.drag_start = res  # (idx, r, c)
            
            # 드래그 사각형 초기화 (선택된 셀에 딱 맞게 시작)
            if self.drag_rect_id: 
                self.canvas.delete(self.drag_rect_id)
            
            rect = self.cell_to_rect_global(*res)
            if rect:
                x1, y1, x2, y2 = rect
                self.drag_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="red", width=2, dash=(4, 4))

    def on_drag(self, e):
        if not self.drag_start: return
        
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        
        # 현재 마우스 위치의 셀 찾기
        current_res = self.xy_to_cell_global(cx, cy)
        
        # 같은 블록 내에 있을 때만 드래그 영역 업데이트
        if current_res and current_res[0] == self.drag_start[0]:
            start_idx, start_r, start_c = self.drag_start
            _, end_r, end_c = current_res
            
            # 시작 셀과 현재 셀을 포함하는 최소/최대 범위 계산
            min_r, max_r = min(start_r, end_r), max(start_r, end_r)
            min_c, max_c = min(start_c, end_c), max(start_c, end_c)
            
            # 픽셀 좌표로 변환 (좌상단 셀의 시작점 ~ 우하단 셀의 끝점)
            rect_start = self.cell_to_rect_global(start_idx, min_r, min_c)
            rect_end = self.cell_to_rect_global(start_idx, max_r, max_c)
            
            if rect_start and rect_end:
                x1, y1, _, _ = rect_start
                _, _, x2, y2 = rect_end
                
                # 사각형 업데이트 (격자에 스냅됨)
                if self.drag_rect_id:
                    self.canvas.coords(self.drag_rect_id, x1, y1, x2, y2)

    def on_release(self, e):
        if self.drag_rect_id:
            self.canvas.delete(self.drag_rect_id)
            self.drag_rect_id = None
        if not self.drag_start: return
        end_cell = self.xy_to_cell(e.x, e.y) or self.drag_start
        r1, c1 = self.drag_start
        r2, c2 = end_cell
        is_color = self.mode_color.get()
        is_number = self.mode_number.get()
        is_eraser = self.mode_eraser.get()
        is_socket = self.mode_socket.get()

        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                pos = (r, c)
                if is_eraser: 
                    self.cell_colors.pop(pos, None); self.cell_numbers.pop(pos, None); self.sockets.discard(pos)
                elif is_socket:
                    self.sockets.add(pos)
                elif is_color or is_number:
                    if is_color: self.cell_colors[pos] = self.current_color.get()
                    if is_number: self.cell_numbers[pos] = parse_number_like(self.current_label.get())
        self.drag_start = None
        self.draw_all()
        self._recompute_all()

    def on_press_right(self, e):
        self._save_state_for_undo()
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        res = self.xy_to_cell_global(cx, cy)
        
        if res:
            self.drag_start = res
            
            if self.drag_rect_id: self.canvas.delete(self.drag_rect_id)
            
            rect = self.cell_to_rect_global(*res)
            if rect:
                x1, y1, x2, y2 = rect
                # 지우개는 파란색 점선
                self.drag_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline="blue", width=2, dash=(4, 4))

    def on_drag_right(self, e):
        # 좌클릭 드래그와 로직 동일 (색상만 다름)
        if not self.drag_start: return
        
        cx = self.canvas.canvasx(e.x)
        cy = self.canvas.canvasy(e.y)
        current_res = self.xy_to_cell_global(cx, cy)
        
        if current_res and current_res[0] == self.drag_start[0]:
            start_idx, start_r, start_c = self.drag_start
            _, end_r, end_c = current_res
            
            min_r, max_r = min(start_r, end_r), max(start_r, end_r)
            min_c, max_c = min(start_c, end_c), max(start_c, end_c)
            
            rect_start = self.cell_to_rect_global(start_idx, min_r, min_c)
            rect_end = self.cell_to_rect_global(start_idx, max_r, max_c)
            
            if rect_start and rect_end:
                x1, y1, _, _ = rect_start
                _, _, x2, y2 = rect_end
                
                if self.drag_rect_id:
                    self.canvas.coords(self.drag_rect_id, x1, y1, x2, y2)

    def on_release_right(self, e):
        if self.drag_rect_id_right:
            self.canvas.delete(self.drag_rect_id_right)
            self.drag_rect_id_right = None
        if not self.drag_start_right: return
        end_cell = self.xy_to_cell(e.x, e.y) or self.drag_start_right
        r1, c1 = self.drag_start_right
        r2, c2 = end_cell
        is_color = self.mode_color.get()
        is_number = self.mode_number.get()
        is_socket = self.mode_socket.get()
        for r in range(min(r1, r2), max(r1, r2) + 1):
            for c in range(min(c1, c2), max(c1, c2) + 1):
                pos = (r, c)
                if is_color: self.cell_colors.pop(pos, None)
                if is_number: self.cell_numbers.pop(pos, None)
                if is_socket: self.sockets.discard(pos)
        self.drag_start_right = None
        self.draw_all()
        self._recompute_all()

    def update_drag_rect(self, e, left_click=True):
        start = self.drag_start if left_click else self.drag_start_right
        if not start: return
        cell = self.xy_to_cell(e.x, e.y)
        if not cell: return
        rect_id = self.drag_rect_id if left_click else self.drag_rect_id_right
        if rect_id: self.canvas.delete(rect_id)
        r1, c1 = start
        r2, c2 = cell
        x1, y1, _, _ = self.cell_to_xy(min(r1, r2), min(c1, c2))
        _, _, x2, y2 = self.cell_to_xy(max(r1, r2), max(c1, c2))
        outline_color = "#3B82F6" if left_click else "#EF4444"
        new_rect_id = self.canvas.create_rectangle(x1, y1, x2, y2, outline=outline_color, width=2, dash=(4, 2))
        if left_click: self.drag_rect_id = new_rect_id
        else: self.drag_rect_id_right = new_rect_id

    def refresh_list(self):
        sel = self.items_list.curselection()
        self.items_list.delete(0, tk.END)
        for i, it in enumerate(self.items, 1):
            self.items_list.insert(tk.END, self.item_label(it, i))
        if sel: self.items_list.selection_set(sel[0])
        self._update_insert_spin_range()
        self._recompute_all()

    def _update_insert_spin_range(self):
        total = len(self.items) + 1

    def item_label(self, it, idx):
        if isinstance(it, SectionHeader): return f"[Section] {it.title}"
        b: Block = it
        label = f"Hatch:{b.hatch}" if b.hatch else (f"Hold:{b.hold}" if b.hold else "-")
        return f"Block {idx} — {b.rows}x{b.cols} | {label} Bay:{b.bay or '-'} Deck:{b.deck or '-'} | Nums:{len(b.cell_numbers)}"

    def _write_gang_counts(self, counts):
        for g in range(3, 10): self.gang_vars[g].set(str(counts.get(g, 0)))

    def edit_selected(self, event=None):
        sel = self.items_list.curselection()
        if not sel:
            messagebox.showinfo("Info", "편집할 블럭을 선택하세요.")
            return
        idx = sel[0]
        it = self.items[idx]
        if isinstance(it, SectionHeader): return
        self.editing_index = idx
        b: Block = it
        self.rows, self.cols = b.rows, b.cols
        self.cell_colors, self.cell_numbers, self.sockets = dict(b.cell_colors), dict(b.cell_numbers), set(b.sockets)
        self._write_gang_counts(b.gang_counts)
        if b.rows >= 6:
            self.meta_label_type.set("Hold No")
            self.meta_no.set(b.hold)
        else:
            self.meta_label_type.set("Hatch No")
            self.meta_no.set(b.hatch)
        self.meta_bay.set(b.bay); self.meta_deck.set(b.deck)
        self.draw_all()
        self.winfo_toplevel().title("Reefer Layout [EDITING]")

    def save_edits(self):
        self._save_state_for_undo()
        if self.editing_index is None: return
        b: Block = self.items[self.editing_index]
        b.cell_colors, b.cell_numbers, b.sockets = dict(self.cell_colors), dict(self.cell_numbers), set(self.sockets)
        b.gang_counts = {g: int(self.gang_vars[g].get() or 0) for g in range(3, 10)}
        self.refresh_list()
        self.editing_index = None
        self.winfo_toplevel().title("Reefer Layout")
        messagebox.showinfo("Saved", "블럭 수정이 저장되었습니다.")

    def _show_context_menu(self, event):
        try:
            index = self.items_list.nearest(event.y)
            if index is not None:
                self.items_list.selection_clear(0, tk.END)
                self.items_list.selection_set(index)
                self.context_menu.tk_popup(event.x_root, event.y_root)
        finally:
            self.context_menu.grab_release()

    def context_menu_action(self, action: str):
        sel = self.items_list.curselection()
        if not sel: 
            if action.startswith("INSERT_"):
                self.insert_section_at(action.replace("INSERT_", ""), pos=len(self.items))
            return
        idx = sel[0]
        if action == "DELETE":
            it = self.items[idx]
            if isinstance(it, SectionHeader):
                 confirm_msg = f"정말로 선택한 섹션({it.title})를 삭제하시겠습니까?"
            else:
                 confirm_msg = "정말로 선택한 블럭을 삭제하시겠습니까?"
            if messagebox.askyesno("Confirm", confirm_msg):
                self._save_state_for_undo()
                del self.items[idx]
                self.refresh_list()
        elif action.startswith("INSERT_"):
            self.insert_section_at(action.replace("INSERT_", ""), pos=idx + 1)

    def apply_meta_to_selected(self):
        sel = self.items_list.curselection()
        if not sel: return
        self._save_state_for_undo()
        b: Block = self.items[sel[0]]
        if self.meta_label_type.get() == "Hold No": b.hold, b.hatch = self.meta_no.get(), ""
        else: b.hatch, b.hold = self.meta_no.get(), ""
        b.bay, b.deck = self.meta_bay.get(), self.meta_deck.get()
        self.refresh_list()

    def insert_section_at(self, title: str, pos: Optional[int] = None):
        self._save_state_for_undo()
        if pos is None:
            try: pos = int(self.insert_pos.get()) - 1
            except (ValueError, TypeError): pos = len(self.items)
        insert_idx = min(max(0, pos), len(self.items))
        self.items.insert(insert_idx, SectionHeader(title=title))
        self.refresh_list()

    def import_from_pdf(self):
        path = filedialog.askopenfilename(
            title="처리할 PDF를 선택하세요",
            filetypes=[("PDF files", "*.pdf;*.PDF")]
        )
        if not path: 
            return
        if not is_valid_pdf(path): 
            messagebox.showerror("오류", "유효한 PDF 파일이 아닙니다.")
            return

        self._save_state_for_undo()
        try:
            doc = fitz.open(path)
            all_rows = []
            
            for i, page in enumerate(doc):
                if page.rotation != 0:
                    page.set_rotation(0)

                words_pdf = extract_words_pdf(page)
                use_ocr = words_pdf.empty or words_pdf["text"].astype(str).str.contains(r"[A-Za-z0-9]", regex=True).sum() < 3
                
                words = ocr_words(page, zoom=5.0) if use_ocr else words_pdf
                tokens = find_R_with_number(words)
                tokens = group_tokens_by_x_and_y(tokens, x_tol=20.0, y_tol=15.0, use_adaptive_tol=True, y_k=1.2)

                if not tokens.empty:
                    tokens.insert(0, "page", i + 1)
                    all_rows.append(tokens)

            if not all_rows:
                messagebox.showinfo("정보", "PDF에서 (R...) 토큰을 찾지 못했습니다.")
                return

            out = pd.concat(all_rows, ignore_index=True)

            self.last_imported_pdf_path = path
            self.last_grouped_tokens = out.copy()
            self.group_to_block_map.clear()

            grouped_result = (
                out.groupby("GroupID")["num"]
                   .apply(lambda x: ",".join(map(str, x)))
                   .reset_index()
            )
            
            current_block_index = len(self.items)
            added, skipped = 0, 0
            default_fill_color = "#EEEEEE" 
            
            for i, row_tuple in enumerate(grouped_result.itertuples()):
                group_id = row_tuple.GroupID
                num_str = row_tuple.num
                
                if not num_str or not num_str.strip():
                    skipped += 1
                    continue
                try:
                    values = [int(p.strip()) for p in num_str.replace("，", ",").split(",") if p.strip()]
                    if not values: raise ValueError("No valid numbers found")
                    rows, cols = len(values), max(values)
                    b = Block(rows=rows, cols=cols)
                    
                    for r_idx, cnt in enumerate(values, 1):
                        start_c = 1 + (cols - cnt) // 2
                        for k in range(cnt): 
                            b.cell_colors[(r_idx, start_c + k)] = default_fill_color
                    
                    self.items.append(b)
                    self.group_to_block_map[group_id] = current_block_index + added
                    added += 1
                except Exception as e:
                    print(f"Skipping group '{num_str}': {e}")
                    skipped += 1

            self.refresh_list()
            messagebox.showinfo("Import PDF", f"완료: {added}개 블럭 추가, {skipped}개 그룹 무시")
        except Exception as e:
            messagebox.showerror("PDF 처리 오류", f"PDF 파일을 처리하는 중 오류가 발생했습니다:\n{e}")

    def export_annotated_pdf(self):
        """이전에 가져온 PDF에 그룹 경계 상자와 Block 번호를 그려 새로운 PDF로 저장합니다."""
        if self.last_grouped_tokens is None or self.last_grouped_tokens.empty:
            messagebox.showinfo("내보내기 오류", "먼저 'Import PDF'를 실행하여 데이터를 가져와야 합니다.")
            return
            
        input_path = self.last_imported_pdf_path
        output_path = filedialog.asksaveasfilename(
            defaultextension=".pdf", 
            filetypes=[("PDF files", "*.pdf")],
            initialfile=f"annotated_{os.path.basename(input_path)}"
        )
        if not output_path: return
        
        try:
            doc = fitz.open(input_path)
            if not self.group_to_block_map:
                messagebox.showerror("PDF 출력 오류", "GroupID-Block 인덱스 맵이 없습니다. 'Import PDF'를 재실행해주세요.")
                return
            
            padding = 7.0
            font_size = 12.0
            text_h = font_size + 2.0
            text_w = 80.0
            margin_above_box = 3.0
            text_color = (0, 0, 0)

            for page_num, page_groups in self.last_grouped_tokens.groupby("page"):
                if page_num > len(doc): continue 
                page = doc[int(page_num) - 1] 
                
                for group_id, group_tokens in page_groups.groupby("GroupID"):
                    if group_id not in self.group_to_block_map: continue 
                        
                    block_idx = self.group_to_block_map[group_id]
                    block_label = f"Block {block_idx + 1}" 

                    min_x0 = group_tokens["x0"].min()
                    min_y0 = group_tokens["y0"].min()
                    max_x1 = group_tokens["x1"].max()
                    max_y1 = group_tokens["y1"].max()
                    
                    new_x0 = min_x0 - padding
                    new_y0 = min_y0 - padding
                    new_x1 = max_x1 + padding
                    new_y1 = max_y1 + padding
                    
                    rect = fitz.Rect(new_x0, new_y0, new_x1, new_y1)
                    page.draw_rect(rect, color=(1, 0, 0), width=2, fill=None)
                    
                    center_x = (new_x0 + new_x1) / 2
                    bottom_y = new_y0 - margin_above_box
                    top_y = bottom_y - text_h
                    
                    insert_rect = fitz.Rect(center_x - text_w / 2, top_y, center_x + text_w / 2, bottom_y)
                    page.insert_textbox(insert_rect, block_label, fontname="helv", fontsize=font_size, color=text_color, align=fitz.TEXT_ALIGN_CENTER)
                        
            doc.save(output_path, garbage=4, clean=True)
            doc.close()
            messagebox.showinfo("내보내기 완료", f"주석이 추가된 PDF 저장 완료:\n{output_path}")
            
        except Exception as e:
            messagebox.showerror("PDF 출력 오류", f"PDF 파일 내보내기 중 오류 발생:\n{e}")

    def save_project(self):
        """현재 작업 상태를 파일로 저장합니다."""
        data = {
            "version": 1.0,
            "items": self.items,
            "ship_no": self.ship_no.get(),
            "rs_count": self.rs_count.get(),
            "rd_per_rs": self.rd_per_rs.get(),
            "last_imported_pdf_path": getattr(self, "last_imported_pdf_path", None),
            "last_grouped_tokens": getattr(self, "last_grouped_tokens", None),
            "group_to_block_map": getattr(self, "group_to_block_map", {})
        }
        file_path = filedialog.asksaveasfilename(
            defaultextension=".dat", 
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
            title="프로젝트 저장"
        )
        if file_path:
            try:
                with open(file_path, "wb") as f:
                    pickle.dump(data, f)
                messagebox.showinfo("저장 완료", "프로젝트가 성공적으로 저장되었습니다.")
            except Exception as e:
                messagebox.showerror("저장 실패", f"파일 저장 중 오류가 발생했습니다:\n{e}")

    def load_project(self):
        """저장된 프로젝트 파일을 불러옵니다."""
        file_path = filedialog.askopenfilename(
            filetypes=[("Data files", "*.dat"), ("All files", "*.*")],
            title="프로젝트 불러오기"
        )
        if not file_path: return

        if messagebox.askyesno("확인", "현재 작업 중인 내용이 사라집니다. 불러오시겠습니까?"):
            try:
                with open(file_path, "rb") as f:
                    data = pickle.load(f)
                
                self.items = data.get("items", [])
                self.ship_no.set(data.get("ship_no", ""))
                self.rs_count.set(data.get("rs_count", "0"))
                self.rd_per_rs.set(data.get("rd_per_rs", "0"))
                
                self.last_imported_pdf_path = data.get("last_imported_pdf_path")
                self.last_grouped_tokens = data.get("last_grouped_tokens")
                self.group_to_block_map = data.get("group_to_block_map", {})

                self.refresh_list()
                self._recompute_all()
                self.clear_canvas()
                self.editing_index = None
                
                messagebox.showinfo("로드 완료", "프로젝트를 성공적으로 불러왔습니다.")
            except Exception as e:
                messagebox.showerror("로드 실패", f"파일을 불러오는 중 오류가 발생했습니다:\n{e}")

    def export_excel(self):
        if not self.items: return
        path = filedialog.asksaveasfilename(defaultextension=".xlsx", filetypes=[("Excel", "*.xlsx")])
        if path:
            try:
                write_excel(self.items, path, self.ship_no.get(), self.looking_txt.get(), self.rs_count.get(), self.rd_per_rs.get())
                messagebox.showinfo("Exported", f"파일 저장 완료:\n{path}")
            except Exception as e:
                messagebox.showerror("Error", f"저장 실패:\n{e}")

    def auto_set_groups(self):
        self._save_state_for_undo()
        sel = self.items_list.curselection()
        targets = [self.items[sel[0]]] if sel and isinstance(self.items[sel[0]], Block) else [it for it in self.items if isinstance(it, Block)]
        if not targets:
            messagebox.showinfo("Auto Set", "대상 블럭이 없습니다.")
            return

        cap = simpledialog.askinteger("Auto Set", "RD panel 당 최대 컨테이너 수", minvalue=4, maxvalue=9999, parent=self)
        if cap is None: return 
        
        try:
            rs_total = int(self.rs_count.get() or 0)
            rpr = int(self.rd_per_rs.get() or 0)
        except(ValueError, TypeError):
            rs_total, rpr = 0, 0
            
        even_list, odd_list = _build_rd_queues(rs_total, rpr)
        all_left_placements, all_right_placements = [], []
        total_left_cells, total_right_cells = 0, 0
        any_failure = False

        for b in targets:
            b.cell_numbers.clear() 
            b.gang_counts = {g: 0 for g in range(3, 10)}
            unfilled_cells = {p for p, color in b.cell_colors.items()}
            center_col = (b.cols + 1) / 2.0
            left_active = {p for p in unfilled_cells if p[1] < center_col}
            right_active = {p for p in unfilled_cells if p[1] > center_col}
            center_cells = {p for p in unfilled_cells if p[1] == center_col}
            
            if len(left_active) <= len(right_active): left_active.update(center_cells)
            else: right_active.update(center_cells)

            placements = [] 
            success = False
            if b.rows >= 6:
                success = self._solve_line_tiling(b, left_active, "LEFT", placements) and self._solve_line_tiling(b, right_active, "RIGHT", placements)
            else:
                right_placements, left_placements = [], []
                if self._solve_tiling_recursive(b, right_active, "RIGHT", right_placements):
                    if self._solve_tiling_recursive(b, left_active, "LEFT", left_placements):
                        placements = right_placements + left_placements
                        success = True

            if not success:
                messagebox.showerror("배치 실패", f"블록 {b.hatch or b.hold or '(번호 없음)'}에서 빈 칸 없이 모든 공간을 채우는 조합을 찾지 못했습니다.\n블록 모양을 확인해주세요.")
                any_failure = True
                continue

            left_placements_b = [p for p in placements if p['side'] == 'LEFT']
            right_placements_b = [p for p in placements if p['side'] == 'RIGHT']
            left_placements_b.sort(key=lambda p: (min(r for r,c in p['cells']), min(c for r,c in p['cells'])))
            right_placements_b.sort(key=lambda p: (min(r for r,c in p['cells']), -max(c for r,c in p['cells'])))

            for p in left_placements_b:
                all_left_placements.append( (b, p) ) 
                total_left_cells += p['size']
            for p in right_placements_b:
                all_right_placements.append( (b, p) ) 
                total_right_cells += p['size']

        if any_failure: return

        target_counts_even, rd_remaining_even = {}, {rd: cap for rd in even_list} 
        total_even_rds = len(even_list)
        if total_even_rds > 0 and total_left_cells > 0:
            avg_cap_even = total_left_cells // total_even_rds 
            rem_even = total_left_cells % total_even_rds     
            for i, rd in enumerate(even_list):
                target_counts_even[rd] = avg_cap_even + (1 if i < rem_even else 0)
        
        target_counts_odd, rd_remaining_odd = {}, {rd: cap for rd in odd_list} 
        total_odd_rds = len(odd_list)
        if total_odd_rds > 0 and total_right_cells > 0:
            avg_cap_odd = total_right_cells // total_odd_rds 
            rem_odd = total_right_cells % total_odd_rds     
            for i, rd in enumerate(odd_list):
                target_counts_odd[rd] = avg_cap_odd + (1 if i < rem_odd else 0)

        cur_even = [0] 
        for (b, p) in all_left_placements: 
            need = p['size'] 
            cells = p['cells'] 
            assigned_rd = self._rd_take_v10(need, target_counts_even, rd_remaining_even, even_list, cur_even)
            if assigned_rd is not None:
                color = self._pick_color(cells, b)
                for cell in cells:
                    b.cell_numbers[cell] = assigned_rd
                    b.cell_colors[cell] = color
                b.gang_counts[need] = b.gang_counts.get(need, 0) + 1

        cur_odd = [0] 
        for (b, p) in all_right_placements: 
            need = p['size'] 
            cells = p['cells'] 
            assigned_rd = self._rd_take_v10(need, target_counts_odd, rd_remaining_odd, odd_list, cur_odd)
            if assigned_rd is not None:
                color = self._pick_color(cells, b)
                for cell in cells:
                    b.cell_numbers[cell] = assigned_rd
                    b.cell_colors[cell] = color
                b.gang_counts[need] = b.gang_counts.get(need, 0) + 1

        if self.editing_index is not None: self.edit_selected() 
        self.refresh_list() 
        self._recompute_all() 
        messagebox.showinfo("Auto Set", "자동 배치 완료")

    def _solve_line_tiling(self, b: Block, side_active: Set[Cell], side: str, placements: List[Dict]) -> bool:
        def partition(length: int) -> Optional[List[int]]:
            sizes = sorted([s for s in SHAPE_LIBRARY.keys() if s <= length], reverse=True)
            memo = {}
            def find(rem_len):
                if rem_len == 0: return []
                if rem_len < 4 : return None
                if rem_len in memo: return memo[rem_len]
                for size in sizes:
                    if size <= rem_len:
                        res = find(rem_len - size)
                        if res is not None:
                            memo[rem_len] = [size] + res
                            return memo[rem_len]
                memo[rem_len] = None
                return None
            return find(length)

        rows = sorted(list({r for r, c in side_active}))
        for r in rows:
            cols_in_row = sorted([c for r_c, c in side_active if r_c == r])
            if not cols_in_row: continue
            spans = []
            start = cols_in_row[0]
            for i in range(1, len(cols_in_row)):
                if cols_in_row[i] != cols_in_row[i-1] + 1:
                    spans.append((start, cols_in_row[i-1]))
                    start = cols_in_row[i]
            spans.append((start, cols_in_row[-1]))
            for start_col, end_col in spans:
                span_len = end_col - start_col + 1
                pieces = partition(span_len)
                if pieces is None: return False
                ptr = start_col
                for size in pieces:
                    cells = {(r, c) for c in range(ptr, ptr + size)}
                    placements.append({'size': size, 'cells': cells, 'side': side})
                    ptr += size
        return True

    def _solve_tiling_recursive(self, b: Block, unfilled: Set[Cell], side: str, placements: List[Dict]) -> bool:
        if not unfilled: return True
        is_left = (side == "LEFT")
        start_cell = min(unfilled, key=lambda p: (p[0], p[1] if is_left else -p[1]))
        priority_sizes = [9, 7, 8, 6, 5, 4]
        for size in priority_sizes:
            for shape_pattern in SHAPE_LIBRARY.get(size, []):
                shape_height = max(r_off for r_off, c_off in shape_pattern) + 1
                shape_width = max(c_off for r_off, c_off in shape_pattern) + 1
                if shape_height == 1 or shape_width == 1: continue 
                group_cells = {(start_cell[0] + r_off, start_cell[1] + (c_off if is_left else -c_off)) for r_off, c_off in shape_pattern}
                if group_cells.issubset(unfilled):
                    placements.append({'size': size, 'cells': group_cells, 'side': side})
                    if self._solve_tiling_recursive(b, unfilled - group_cells, side, placements): return True
                    placements.pop()

        for size in priority_sizes:
            for shape_pattern in SHAPE_LIBRARY.get(size, []):
                shape_height = max(r_off for r_off, c_off in shape_pattern) + 1
                shape_width = max(c_off for r_off, c_off in shape_pattern) + 1
                if not (shape_height == 1 or shape_width == 1): continue
                group_cells = {(start_cell[0] + r_off, start_cell[1] + (c_off if is_left else -c_off)) for r_off, c_off in shape_pattern}
                if not group_cells.issubset(unfilled): continue
                remaining_unfilled = unfilled - group_cells
                is_isolated = True
                for r_cell, c_cell in group_cells:
                    if (r_cell - 1, c_cell) in remaining_unfilled or (r_cell + 1, c_cell) in remaining_unfilled:
                        is_isolated = False; break
                if not is_isolated: continue
                placements.append({'size': size, 'cells': group_cells, 'side': side})
                if self._solve_tiling_recursive(b, unfilled - group_cells, side, placements): return True
                placements.pop()
        return False

    def _rd_take_v10(self, need, target_counts, rd_remaining, rds, cursor):
        n = len(rds)
        if n == 0: return None

        current_idx = cursor[0] 
        while current_idx < n:
            current_rd = rds[current_idx]
            
            can_fit = rd_remaining.get(current_rd, 0) >= need
            below_target = target_counts.get(current_rd, 0) > 0 
        
            if can_fit and below_target:
                rd_remaining[current_rd] -= need
                target_counts[current_rd] = target_counts.get(current_rd, 0) - need 
                cursor[0] = current_idx
                return current_rd
            current_idx += 1 
            
        for i in range(n): 
            check_idx = i 
            check_rd = rds[check_idx]
            if rd_remaining.get(check_rd, 0) >= need:
                rd_remaining[check_rd] -= need
                target_counts[check_rd] = target_counts.get(check_rd, 0) - need 
                cursor[0] = check_idx
                return check_rd
        return None

    def _pick_color(self, cells, b):
        palette = self.FIXED_COLORS 
        adj_colors = set()
        for r, c in cells:
            for dr, dc in [(0, 1), (0, -1), (1, 0), (-1, 0)]: 
                adj_cell = (r + dr, c + dc)
                if adj_cell in b.cell_colors: 
                    adj_colors.add(b.cell_colors[adj_cell]) 
        
        for color in palette:
            if color not in adj_colors:
                return color
        return palette[0]

class App(tk.Tk):
    def __init__(self):
        super().__init__()
        self.title("Reefer Layout")
        try:
            from ctypes import windll; windll.shcore.SetProcessDpiAwareness(1)
        except: pass
        self.geometry("1200x800") 
        self.minsize(1200, 800) 
        GridCanvas(self) 

if __name__ == '__main__':
    app = App() 
    app.mainloop()
