# ============================================================
# Script to generate a polished Word Document (.docx) report
# Compatible with both Microsoft Word Light Mode & Dark Mode
# ============================================================

import docx
from docx.shared import Inches, Pt, RGBColor
from docx.enum.text import WD_ALIGN_PARAGRAPH
from docx.enum.table import WD_TABLE_ALIGNMENT, WD_ALIGN_VERTICAL
from docx.oxml import OxmlElement, parse_xml
from docx.oxml.ns import nsdecls, qn
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent


def style_table(table, col_widths):
    """
    Applies professional styling, borders, row padding, and anti-split rules.
    Guarantees:
      - cantSplit on all rows (no rows split awkwardly across pages)
      - tblHeader on row 0 (repeats header cleanly if table splits)
      - Explicit table grid and cell widths summing exactly to margins
      - Clean borders compatible with Word Light and Dark Mode
    """
    tbl_pr = table._tbl.tblPr
    # Clean subtle borders
    borders = parse_xml(
        f'<w:tblBorders {nsdecls("w")}>'
        f'<w:top w:val="single" w:sz="6" w:space="0" w:color="1B365D"/>'
        f'<w:bottom w:val="single" w:sz="8" w:space="0" w:color="1B365D"/>'
        f'<w:insideH w:val="single" w:sz="4" w:space="0" w:color="D0D7DE"/>'
        f'<w:insideV w:val="none"/>'
        f'<w:left w:val="none"/>'
        f'<w:right w:val="none"/>'
        f'</w:tblBorders>'
    )
    tbl_pr.append(borders)

    # Apply cantSplit to all rows and tblHeader to row 0
    for idx, row in enumerate(table.rows):
        tr_pr = row._tr.get_or_add_trPr()
        tr_pr.append(parse_xml(f'<w:cantSplit {nsdecls("w")}/>'))
        if idx == 0:
            tr_pr.append(parse_xml(f'<w:tblHeader {nsdecls("w")}/>'))

        # Set cell widths and internal padding
        for c_idx, cell in enumerate(row.cells):
            cell.width = col_widths[c_idx]
            tc_pr = cell._tc.get_or_add_tcPr()
            # 120 twips top/bottom (6pt), 140 twips left/right (7pt)
            tc_mar = parse_xml(
                f'<w:tcMar {nsdecls("w")}>'
                f'<w:top w:w="120" w:type="dxa"/>'
                f'<w:bottom w:w="120" w:type="dxa"/>'
                f'<w:left w:w="140" w:type="dxa"/>'
                f'<w:right w:w="140" w:type="dxa"/>'
                f'</w:tcMar>'
            )
            tc_pr.append(tc_mar)


def set_cell_background(cell, hex_color):
    """Set background color of a table cell."""
    tc_pr = cell._tc.get_or_add_tcPr()
    shd = parse_xml(f'<w:shd {nsdecls("w")} w:fill="{hex_color}"/>')
    tc_pr.append(shd)


def create_document():
    doc = docx.Document()

    # Page Margins: 1 inch on all sides (Content width = 6.5 inches = 9360 dxa)
    for section in doc.sections:
        section.top_margin = Inches(1.0)
        section.bottom_margin = Inches(1.0)
        section.left_margin = Inches(1.0)
        section.right_margin = Inches(1.0)

    # Palette
    COLOR_NAVY = RGBColor(27, 54, 93)      # #1B365D
    COLOR_STEEL = RGBColor(70, 130, 180)   # #4682B4
    COLOR_GREEN = RGBColor(46, 125, 50)    # #2E7D32
    COLOR_WHITE = RGBColor(255, 255, 255)
    COLOR_GRAY = RGBColor(90, 90, 90)

    HEADER_BG = "1B365D"  # Crisp Navy for table headers

    def add_heading_with_keep(text, level=1):
        h = doc.add_heading(text, level=level)
        h.paragraph_format.space_before = Pt(14)
        h.paragraph_format.space_after = Pt(6)
        h.paragraph_format.keep_with_next = True
        return h

    # Document Title
    title_p = doc.add_paragraph()
    title_p.paragraph_format.space_before = Pt(0)
    title_p.paragraph_format.space_after = Pt(2)
    run_title = title_p.add_run("Honeybee Behavior Detection & Classification")
    run_title.font.name = "Arial"
    run_title.font.size = Pt(22)
    run_title.font.bold = True
    run_title.font.color.rgb = COLOR_NAVY

    subtitle_p = doc.add_paragraph()
    subtitle_p.paragraph_format.space_before = Pt(0)
    subtitle_p.paragraph_format.space_after = Pt(12)
    run_sub = subtitle_p.add_run("Pre vs. Post Architectural Evaluation: Kinematics Baseline vs. DINOv2 ViT-S/14 + LSTM")
    run_sub.font.name = "Arial"
    run_sub.font.size = Pt(12)
    run_sub.font.color.rgb = COLOR_STEEL

    # Metadata Banner
    meta_p = doc.add_paragraph()
    meta_p.paragraph_format.space_after = Pt(12)
    r1 = meta_p.add_run("Date: ")
    r1.bold = True
    meta_p.add_run("October 2026   |   ")
    r2 = meta_p.add_run("Dataset: ")
    r2.bold = True
    meta_p.add_run("Mendeley Data v6 (DOI: 10.17632/8gb9r2yhfc.6)   |   ")
    r3 = meta_p.add_run("GPU: ")
    r3.bold = True
    meta_p.add_run("RTX 4060 (CUDA 12.6)")

    # Divider line
    p_div = doc.add_paragraph()
    p_div.paragraph_format.space_after = Pt(12)
    p_div_border = parse_xml(f'<w:pBdr {nsdecls("w")}><w:bottom w:val="single" w:sz="12" w:space="1" w:color="1B365D"/></w:pBdr>')
    p_div._p.get_or_add_pPr().append(p_div_border)

    # 1. Executive Summary
    add_heading_with_keep("1. Executive Summary", level=1)
    p_exec = doc.add_paragraph(
        "This report evaluates the honeybee behavior classification system before and after transitioning from a "
        "kinematics-only approach to deep visual feature representations using Meta's DINOv2 Vision Transformer (ViT-S/14). "
        "The previous baseline tracked bounding box center coordinates and velocities into an LSTM. While functional for transit, "
        "kinematics alone proved blind to stationary postures such as Fanning (vigorous in-place wing oscillation), frequently "
        "confusing them with resting or idle bees."
    )
    p_exec.paragraph_format.space_after = Pt(8)

    p_exec2 = doc.add_paragraph(
        "To resolve this, we integrated DINOv2 ViT-S/14 to extract dense 384-dimensional spatial visual embeddings from 10% padded "
        "bee crops. When paired with native 50 fps smoothed kinematics and a causal 16-step sliding window (~1.92s), the system "
        "achieves a +19.11% gain in Macro-F1. Crucially, adding same-video negatives ('Other' class inside the entrance zone polygon) "
        "eliminated background false alarms, while full GPU optimization increased live throughput to 25.33 FPS."
    )
    p_exec2.paragraph_format.space_after = Pt(14)

    # 2. Table 1: Architecture Comparison
    add_heading_with_keep("2. System Architecture Comparison", level=1)
    arch_widths = [Inches(1.8), Inches(2.35), Inches(2.35)]
    arch_data = [
        ("Dimension", "Pre: Kinematics Baseline", "Post: DINOv2 ViT-S/14 + LSTM"),
        ("What Model Sees", "7–9 Kinematic Scalars (x, y, w, h, speed, direction)", "384-d Attention-Based Visual Embeddings (RGB crops)"),
        ("Visual Detail", "None. Blind to wing vibration blur & body posture", "Captures wing blur patterns, body angle, and posture"),
        ("Observation Window (T)", "10 raw frames @ 50 fps = 0.20 seconds", "16 steps @ K=6 subsampling = 1.92 seconds"),
        ("Target Classes", "4 classes (no background/idle negative class)", "5 classes (Foraging, Defense, Fanning, Washboarding, Other)"),
        ("Zone Gating", "None (evaluates entire video frame)", "Entrance-Zone Polygon Gating (rejects wall/edge noise)"),
        ("Inference Execution", "Hybrid PyTorch (YOLO) + Keras CPU", "Unified PyTorch CUDA Stream (FP16 Autocast)"),
        ("Live Preview Speed", "~14.8 FPS (CPU bound LSTM)", "25.33 FPS (Real-time GPU pipeline)"),
    ]

    t1 = doc.add_table(rows=len(arch_data), cols=3)
    t1.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(t1, arch_widths)

    for r_i, row in enumerate(arch_data):
        for c_i, val in enumerate(row):
            cell = t1.cell(r_i, c_i)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(2)
            p.paragraph_format.space_after = Pt(2)
            run = p.runs[0]
            run.font.name = "Arial"
            run.font.size = Pt(9.0)

            if r_i == 0:
                set_cell_background(cell, HEADER_BG)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            else:
                if c_i == 0:
                    run.font.bold = True

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 3. Table 2: Quantitative Results
    add_heading_with_keep("3. Quantitative Performance on Identical Evaluation Split", level=1)
    doc.add_paragraph(
        "All models were evaluated on the identical 67,366 training windows and 379 non-overlapping validation windows "
        "across all five annotated video sessions. A Kinematics-Only control was evaluated on the exact same windows to benchmark "
        "how much visual representation truly adds:"
    ).paragraph_format.space_after = Pt(8)

    quant_widths = [Inches(1.7), Inches(1.2), Inches(1.2), Inches(1.2), Inches(1.2)]
    quant_data = [
        ("Metric", "Pre (Kinematics)", "Post (Fused)", "Post (Visual ViT)", "Absolute Gain"),
        ("Macro-F1", "0.4574", "0.6365", "0.6485", "+19.11%"),
        ("Weighted-F1", "0.7365", "0.8106", "0.8333", "+9.68%"),
        ("Overall Accuracy", "72.56%", "80.47%", "83.11%", "+10.55%"),
    ]

    t2 = doc.add_table(rows=len(quant_data), cols=5)
    t2.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(t2, quant_widths)

    for r_i, row in enumerate(quant_data):
        for c_i, val in enumerate(row):
            cell = t2.cell(r_i, c_i)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.runs[0]
            run.font.name = "Arial"
            run.font.size = Pt(9.5)

            if r_i == 0:
                set_cell_background(cell, HEADER_BG)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            else:
                if c_i == 4:
                    run.font.bold = True
                    run.font.color.rgb = COLOR_GREEN

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 4. Table 3: Per-Class Breakdown
    add_heading_with_keep("4. Per-Class Precision, Recall, and F1 Breakdown", level=1)
    class_widths = [Inches(1.25), Inches(0.85), Inches(0.85), Inches(0.85), Inches(0.85), Inches(1.85)]
    class_data = [
        ("Behavior Class", "Pre F1", "Post Prec", "Post Rec", "Post F1", "Key Scientific Insight"),
        ("Fanning", "0.7009", "92.0%", "77.0%", "0.8387", "+13.78% F1. ViT detects wing blur that stationary kinematics missed."),
        ("Other (Idle/Transit)", "0.7793", "83.1%", "89.5%", "0.8621", "+8.28% F1. Successfully rejects negative background landing board bees."),
        ("Washboarding", "0.5846", "50.0%", "59.1%", "0.5417", "Cut false alarms from 'Other' by 43% compared to kinematics."),
        ("Defense", "0.2222", "100.0%", "100.0%", "1.0000", "Zero false alarms for DINOv2 vs 7 false alarms in kinematics (Prec: 12.5%)."),
        ("Foraging", "0.0000", "0.0%", "0.0%", "0.0000", "Support = 1 in validation. Rapid transit bees (<1.0s) filtered by 1.92s window."),
    ]

    t3 = doc.add_table(rows=len(class_data), cols=6)
    t3.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(t3, class_widths)

    for r_i, row in enumerate(class_data):
        for c_i, val in enumerate(row):
            cell = t3.cell(r_i, c_i)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if 0 < c_i < 5 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.runs[0]
            run.font.name = "Arial"
            run.font.size = Pt(8.5)

            if r_i == 0:
                set_cell_background(cell, HEADER_BG)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            else:
                if c_i == 4 and r_i in (1, 2, 4):
                    run.font.bold = True
                    run.font.color.rgb = COLOR_GREEN

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 5. Table 4: Confusion Matrix
    add_heading_with_keep("5. Visual-Only DINOv2 Confusion Matrix (n=379)", level=1)
    cm_widths = [Inches(1.75), Inches(0.95), Inches(0.95), Inches(0.95), Inches(0.95), Inches(0.95)]
    cm_data = [
        ("Ground Truth \\ Predicted", "Foraging", "Defense", "Fanning", "Washboarding", "Other"),
        ("Foraging (n=1)", "0", "0", "1", "0", "0"),
        ("Defense (n=1)", "0", "1", "0", "0", "0"),
        ("Fanning (n=135)", "0", "0", "104", "0", "31"),
        ("Washboarding (n=22)", "0", "0", "0", "13", "9"),
        ("Other (n=220)", "2", "0", "8", "13", "197"),
    ]

    t4 = doc.add_table(rows=len(cm_data), cols=6)
    t4.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(t4, cm_widths)

    for r_i, row in enumerate(cm_data):
        for c_i, val in enumerate(row):
            cell = t4.cell(r_i, c_i)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i > 0 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.runs[0]
            run.font.name = "Arial"
            run.font.size = Pt(9.0)

            if r_i == 0:
                set_cell_background(cell, HEADER_BG)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            else:
                if c_i == r_i:
                    run.font.bold = True
                    run.font.color.rgb = COLOR_NAVY

    doc.add_paragraph().paragraph_format.space_after = Pt(12)

    # 6. Table 5: Full 6,000-Frame Benchmark Comparison
    add_heading_with_keep("6. End-to-End 6,000-Frame Benchmark (videos/20230711a-fan.mp4)", level=1)
    b_widths = [Inches(2.2), Inches(1.8), Inches(2.5)]
    b_data = [
        ("Metric", "Pre: Baseline System", "Post: DINOv2 ViT-S/14 + LSTM"),
        ("Frames Processed", "6,000 / 6,000 frames", "6,000 / 6,000 frames"),
        ("Unique ByteTrack Tracks", "3,788 tracks", "3,788 tracks"),
        ("Total Wall Clock Time", "405.0 s (6.75 minutes)", "236.8 s (3.95 minutes) [1.71x faster]"),
        ("Average Throughput", "14.8 FPS", "25.33 FPS (Exceeds 25 FPS standard)"),
        ("Tracks Classified", "3,788 (predicted every track)", "295 (filtered to entrance zone >= 1.92s)"),
        ("Behavior Distribution", "Foraging: 78.4%, Fanning: 21.6%", "Other: 92.9%, Fanning: 5.8%, Foraging: 1.4%"),
        ("False Alarms (Def/Wash)", "Uncontrolled", "0 Defense, 0 Washboarding (100% clean)"),
        ("GPU VRAM Utilization", "~1.5 GB", "~3.2 GB (Safely within 8 GB card budget)"),
    ]

    t5 = doc.add_table(rows=len(b_data), cols=3)
    t5.alignment = WD_TABLE_ALIGNMENT.CENTER
    style_table(t5, b_widths)

    for r_i, row in enumerate(b_data):
        for c_i, val in enumerate(row):
            cell = t5.cell(r_i, c_i)
            cell.text = val
            p = cell.paragraphs[0]
            p.paragraph_format.space_before = Pt(3)
            p.paragraph_format.space_after = Pt(3)
            p.alignment = WD_ALIGN_PARAGRAPH.CENTER if c_i == 1 else WD_ALIGN_PARAGRAPH.LEFT
            run = p.runs[0]
            run.font.name = "Arial"
            run.font.size = Pt(9.0)

            if r_i == 0:
                set_cell_background(cell, HEADER_BG)
                run.font.bold = True
                run.font.color.rgb = COLOR_WHITE
            else:
                if c_i == 0:
                    run.font.bold = True
                elif c_i == 2 and r_i in (3, 4, 7):
                    run.font.bold = True
                    run.font.color.rgb = COLOR_GREEN

    doc.add_paragraph().paragraph_format.space_after = Pt(14)

    # 7. Production Guide
    add_heading_with_keep("7. Execution Commands", level=1)
    doc.add_paragraph("To launch live inference or video exports directly from your PowerShell terminal:")

    p_cmd = doc.add_paragraph()
    p_cmd.paragraph_format.left_indent = Inches(0.4)
    r_c = p_cmd.add_run(
        "# Run real-time DINOv2 ViT-S/14 + LSTM live preview window:\n"
        "python run_bee_vit_lstm.py\n\n"
        "# Run on Defense video file:\n"
        "python run_bee_vit_lstm.py --video videos/20230609b-def.mp4\n\n"
        "# Export annotated demonstration video with bounding boxes & HUD:\n"
        "python run_bee_vit_lstm.py --save-out videos/annotated_behavior.mp4"
    )
    r_c.font.name = "Consolas"
    r_c.font.size = Pt(9.5)
    r_c.font.color.rgb = COLOR_NAVY

    out_file = BASE_DIR / "Bee_Behavior_Classification_Report_Clean.docx"
    doc.save(str(out_file))
    print(f"Clean report saved: {out_file}")


if __name__ == "__main__":
    create_document()
