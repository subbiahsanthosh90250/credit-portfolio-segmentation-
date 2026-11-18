import os
import re
from docx import Document
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn

BASE_DIR = r"C:\Users\Subbiah Santhosh .RM\Downloads\Project 3\credit_portfolio_segmentation"

# Palette
COLOR_NAVY = RGBColor(30, 61, 89)      # #1E3D59
COLOR_SLATE = RGBColor(70, 80, 95)     # #46505F
COLOR_GOLD = RGBColor(245, 139, 84)    # #F58B54
COLOR_CHARCOAL = RGBColor(33, 37, 41)  # #212529

def set_cell_background(cell, hex_color):
    shading_elm = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    cell._tc.get_or_add_tcPr().append(shading_elm)

def set_cell_margins(cell, top=100, bottom=100, left=150, right=150):
    tcPr = cell._tc.get_or_add_tcPr()
    tcMar = OxmlElement('w:tcMar')
    for m, val in [('top', top), ('bottom', bottom), ('left', left), ('right', right)]:
        node = OxmlElement(f'w:{m}')
        node.set(qn('w:w'), str(val))
        node.set(qn('w:type'), 'dxa')
        tcMar.append(node)
    tcPr.append(tcMar)

def apply_run_formatting(p, text):
    # Splits text by markdown tokens: **bold**, `code`, *italic*
    tokens = re.split(r'(\*\*.*?\*\*|`.*?`|\*.*?\*)', text)
    for token in tokens:
        if not token:
            continue
        if token.startswith('**') and token.endswith('**'):
            run = p.add_run(token[2:-2])
            run.bold = True
            run.font.color.rgb = COLOR_CHARCOAL
        elif token.startswith('`') and token.endswith('`'):
            run = p.add_run(token[1:-1])
            run.font.name = 'Consolas'
            run.font.size = Pt(9.5)
            run.font.color.rgb = RGBColor(180, 40, 40)
        elif token.startswith('*') and token.endswith('*'):
            run = p.add_run(token[1:-1])
            run.italic = True
            run.font.color.rgb = COLOR_SLATE
        else:
            run = p.add_run(token)
            run.font.color.rgb = COLOR_CHARCOAL

def md_to_docx(md_path, docx_path, doc_title):
    doc = Document()
    
    # Page Margins: 1 inch all around
    sections = doc.sections
    for section in sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Base Normal Style
    normal_style = doc.styles['Normal']
    normal_style.font.name = 'Segoe UI'
    normal_style.font.size = Pt(10.5)
    normal_style.font.color.rgb = COLOR_CHARCOAL

    with open(md_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    in_code_block = False
    code_lines = []
    in_table = False
    table_lines = []

    def flush_table():
        nonlocal in_table, table_lines
        if not table_lines:
            return
        
        # Parse table rows
        rows_data = []
        for tl in table_lines:
            cleaned = tl.strip()
            if cleaned.startswith('|') and cleaned.endswith('|'):
                cleaned = cleaned[1:-1]
            parts = [p.strip() for p in cleaned.split('|')]
            # check if separator row
            if all(re.match(r'^:?-+:?$', p) for p in parts if p):
                continue
            rows_data.append(parts)
        
        if not rows_data:
            table_lines = []
            in_table = False
            return
        
        num_cols = max(len(r) for r in rows_data)
        table = doc.add_table(rows=len(rows_data), cols=num_cols)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        
        for r_idx, row in enumerate(rows_data):
            for c_idx in range(num_cols):
                cell_text = row[c_idx] if c_idx < len(row) else ""
                cell = table.cell(r_idx, c_idx)
                p = cell.paragraphs[0]
                p.paragraph_format.space_before = Pt(3)
                p.paragraph_format.space_after = Pt(3)
                
                if r_idx == 0:
                    set_cell_background(cell, "1E3D59")
                    run = p.add_run(cell_text)
                    run.bold = True
                    run.font.color.rgb = RGBColor(255, 255, 255)
                    run.font.size = Pt(9.5)
                else:
                    if r_idx % 2 == 1:
                        set_cell_background(cell, "F8F9FA")
                    else:
                        set_cell_background(cell, "FFFFFF")
                    apply_run_formatting(p, cell_text)
                    for r in p.runs:
                        r.font.size = Pt(9.5)
                set_cell_margins(cell, top=80, bottom=80, left=120, right=120)

        # Space after table
        empty_p = doc.add_paragraph()
        empty_p.paragraph_format.space_before = Pt(0)
        empty_p.paragraph_format.space_after = Pt(6)
        
        table_lines = []
        in_table = False

    def flush_code():
        nonlocal in_code_block, code_lines
        if not code_lines:
            return
        
        table = doc.add_table(rows=1, cols=1)
        table.alignment = WD_TABLE_ALIGNMENT.CENTER
        cell = table.cell(0, 0)
        set_cell_background(cell, "F4F5F7")
        set_cell_margins(cell, top=100, bottom=100, left=150, right=150)
        
        p = cell.paragraphs[0]
        p.paragraph_format.space_before = Pt(2)
        p.paragraph_format.space_after = Pt(2)
        
        full_code = "".join(code_lines)
        run = p.add_run(full_code.rstrip())
        run.font.name = 'Consolas'
        run.font.size = Pt(9)
        run.font.color.rgb = RGBColor(40, 40, 40)
        
        empty_p = doc.add_paragraph()
        empty_p.paragraph_format.space_before = Pt(0)
        empty_p.paragraph_format.space_after = Pt(6)

        code_lines = []
        in_code_block = False

    for line in lines:
        stripped = line.strip()

        # Handle Code Blocks ```
        if stripped.startswith('```'):
            if in_code_block:
                flush_code()
            else:
                if in_table:
                    flush_table()
                in_code_block = True
                code_lines = []
            continue

        if in_code_block:
            code_lines.append(line)
            continue

        # Handle Tables (| col1 | col2 |)
        if stripped.startswith('|') and stripped.endswith('|'):
            in_table = True
            table_lines.append(line)
            continue
        elif in_table:
            flush_table()

        # Blank Line
        if not stripped:
            continue

        # Horizontal Rule
        if re.match(r'^-{3,}$', stripped) or re.match(r'^\*{3,}$', stripped):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(6)
            p.paragraph_format.space_after = Pt(6)
            p_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="8" w:space="1" w:color="CCCCCC"/></w:pBdr>')
            p._p.get_or_add_pPr().append(p_border)
            continue

        # Headings
        if stripped.startswith('# '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(14)
            p.paragraph_format.space_after = Pt(6)
            run = p.add_run(stripped[2:])
            run.bold = True
            run.font.size = Pt(20)
            run.font.color.rgb = COLOR_NAVY
        elif stripped.startswith('## '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(12)
            p.paragraph_format.space_after = Pt(4)
            run = p.add_run(stripped[3:])
            run.bold = True
            run.font.size = Pt(15)
            run.font.color.rgb = COLOR_NAVY
        elif stripped.startswith('### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(10)
            p.paragraph_format.space_after = Pt(3)
            run = p.add_run(stripped[4:])
            run.bold = True
            run.font.size = Pt(12.5)
            run.font.color.rgb = COLOR_NAVY
        elif stripped.startswith('#### '):
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(8)
            p.paragraph_format.space_after = Pt(2)
            run = p.add_run(stripped[5:])
            run.bold = True
            run.font.size = Pt(11)
            run.font.color.rgb = COLOR_GOLD
        # Blockquotes
        elif stripped.startswith('> '):
            p = doc.add_paragraph()
            p.paragraph_format.left_indent = Inches(0.3)
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            # Left border
            p_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:left w:val="single" w:sz="24" w:space="10" w:color="1E3D59"/></w:pBdr>')
            p._p.get_or_add_pPr().append(p_border)
            apply_run_formatting(p, stripped[2:])
            for r in p.runs:
                r.italic = True
                r.font.color.rgb = COLOR_SLATE
        # Bullet Lists
        elif stripped.startswith('- ') or stripped.startswith('* '):
            p = doc.add_paragraph(style='List Bullet')
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            apply_run_formatting(p, stripped[2:])
        # Numbered Lists
        elif re.match(r'^\d+\.\s', stripped):
            text_part = re.sub(r'^\d+\.\s', '', stripped)
            p = doc.add_paragraph(style='List Number')
            p.paragraph_format.space_before = Pt(1.5)
            p.paragraph_format.space_after = Pt(1.5)
            apply_run_formatting(p, text_part)
        # Normal Paragraph
        else:
            p = doc.add_paragraph()
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            apply_run_formatting(p, stripped)

    # In case file ended inside a block
    if in_table:
        flush_table()
    if in_code_block:
        flush_code()

    doc.save(docx_path)
    print(f"Created: {docx_path}")

def convert_all():
    files = [
        ("PROJECT_DOCUMENTATION.md", "PROJECT_DOCUMENTATION.docx", "Project Documentation"),
        ("PRESENTATION_SCRIPT.md", "PRESENTATION_SCRIPT.docx", "Live Presentation Script"),
        ("HOW_TO_RUN.md", "HOW_TO_RUN.docx", "Execution Guide"),
        ("README.md", "README.docx", "Repository Readme")
    ]
    for md_file, docx_file, title in files:
        src = os.path.join(BASE_DIR, md_file)
        dst = os.path.join(BASE_DIR, docx_file)
        if os.path.exists(src):
            md_to_docx(src, dst, title)
        else:
            print(f"Source not found: {src}")

if __name__ == "__main__":
    convert_all()
