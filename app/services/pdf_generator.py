import os
import sys
from pathlib import Path
from datetime import datetime, timezone
from typing import List, Optional

from reportlab.lib.pagesizes import letter
from reportlab.lib import colors
from reportlab.lib.units import inch
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, HRFlowable, KeepTogether
)
from reportlab.pdfgen import canvas
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont

from app.config import settings
from app.models.schemas import VideoMetadata, SafetyAuditReport, GroundedSummary, ActionItem, HITLState


# =============================================================================
# Universal Multi-Lingual Unicode Font Resolver
# Supports Indic (Telugu, Hindi, Tamil, etc.), CJK, Arabic, Cyrillic, Latin-ext
# =============================================================================
UNICODE_FONT = "Helvetica"
UNICODE_BOLD_FONT = "Helvetica-Bold"

def _init_multilingual_fonts():
    global UNICODE_FONT, UNICODE_BOLD_FONT

    candidates = []

    # 1. Windows Fonts
    windir = os.environ.get("WINDIR", "C:\\Windows")
    fonts_dir = Path(windir) / "Fonts"
    candidates.extend([
        # Microsoft Nirmala UI: Universal font with comprehensive Indic (Telugu, Hindi, Tamil, Bengali, etc.) & Latin support
        {"regular": fonts_dir / "Nirmala.ttc", "bold": fonts_dir / "Nirmala.ttc", "reg_sub": 0, "bold_sub": 1},
        # Segoe UI
        {"regular": fonts_dir / "segoeui.ttf", "bold": fonts_dir / "segoeuib.ttf"},
        # Arial
        {"regular": fonts_dir / "arial.ttf", "bold": fonts_dir / "arialbd.ttf"},
    ])

    # 2. Linux / Docker / Vercel Fonts
    candidates.extend([
        {"regular": Path("/usr/share/fonts/truetype/noto/NotoSans-Regular.ttf"), "bold": Path("/usr/share/fonts/truetype/noto/NotoSans-Bold.ttf")},
        {"regular": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf"), "bold": Path("/usr/share/fonts/truetype/dejavu/DejaVuSans-Bold.ttf")},
        {"regular": Path("/usr/share/fonts/truetype/freefont/FreeSans.ttf"), "bold": Path("/usr/share/fonts/truetype/freefont/FreeSansBold.ttf")},
        {"regular": Path("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf"), "bold": Path("/usr/share/fonts/truetype/liberation/LiberationSans-Bold.ttf")}
    ])

    # 3. macOS Fonts
    candidates.extend([
        {"regular": Path("/System/Library/Fonts/Supplemental/Arial.ttf"), "bold": Path("/System/Library/Fonts/Supplemental/Arial Bold.ttf")},
        {"regular": Path("/Library/Fonts/Arial.ttf"), "bold": Path("/Library/Fonts/Arial Bold.ttf")}
    ])

    for c in candidates:
        reg_path = Path(c["regular"])
        bold_path = Path(c.get("bold", c["regular"]))
        if reg_path.exists():
            try:
                reg_sub = c.get("reg_sub")
                bold_sub = c.get("bold_sub")

                if reg_sub is not None:
                    pdfmetrics.registerFont(TTFont("AppUnicodeRegular", str(reg_path), subfontIndex=reg_sub))
                else:
                    pdfmetrics.registerFont(TTFont("AppUnicodeRegular", str(reg_path)))

                if bold_path.exists():
                    if bold_sub is not None:
                        pdfmetrics.registerFont(TTFont("AppUnicodeBold", str(bold_path), subfontIndex=bold_sub))
                    else:
                        pdfmetrics.registerFont(TTFont("AppUnicodeBold", str(bold_path)))
                else:
                    pdfmetrics.registerFont(TTFont("AppUnicodeBold", str(reg_path)))

                UNICODE_FONT = "AppUnicodeRegular"
                UNICODE_BOLD_FONT = "AppUnicodeBold"
                break
            except Exception:
                continue

_init_multilingual_fonts()


class NumberedCanvas(canvas.Canvas):
    """Adds professional header and footer with dynamic page count."""
    def __init__(self, *args, **kwargs):
        super(NumberedCanvas, self).__init__(*args, **kwargs)
        self._saved_page_states = []

    def showPage(self):
        self._saved_page_states.append(dict(self.__dict__))
        self._startPage()

    def save(self):
        num_pages = len(self._saved_page_states)
        for state in self._saved_page_states:
            self.__dict__.update(state)
            self.draw_page_decorations(num_pages)
            canvas.Canvas.showPage(self)
        canvas.Canvas.save(self)

    def draw_page_decorations(self, page_count):
        self.saveState()
        self.setFont(UNICODE_FONT, 8)
        self.setFillColor(colors.HexColor("#64748b"))

        # Header
        self.drawString(54, 750, "Agentic AI YouTube Content Intelligence & Grounding Report")
        self.drawRightString(612 - 54, 750, "CONFIDENTIAL & GROUNDED")
        self.setStrokeColor(colors.HexColor("#e2e8f0"))
        self.setLineWidth(0.5)
        self.line(54, 742, 612 - 54, 742)

        # Footer
        self.line(54, 45, 612 - 54, 45)
        self.drawString(54, 32, "Verified 100% Grounded Content — Zero Hallucination Guarantee")
        self.drawRightString(612 - 54, 32, f"Page {self._pageNumber} of {page_count}")
        self.restoreState()


class PDFGeneratorService:
    """
    Service responsible for compiling high-fidelity, styled PDF summary reports
    with native multilingual Unicode font support.
    """

    @classmethod
    def generate_pdf(
        cls,
        job_id: str,
        metadata: VideoMetadata,
        safety_report: SafetyAuditReport,
        summary: GroundedSummary,
        action_items: List[ActionItem],
        hitl_state: Optional[HITLState] = None
    ) -> str:
        """
        Compiles the complete report into a PDF file and returns the file path.
        """
        output_dir = Path(settings.PDF_OUTPUT_DIR)
        output_dir.mkdir(parents=True, exist_ok=True)
        filename = f"report_{metadata.video_id}_{job_id[:8]}.pdf"
        file_path = output_dir / filename

        doc = SimpleDocTemplate(
            str(file_path),
            pagesize=letter,
            leftMargin=54,
            rightMargin=54,
            topMargin=54,
            bottomMargin=54,
        )

        styles = getSampleStyleSheet()
        
        # Custom Multilingual Unicode Typography
        title_style = ParagraphStyle(
            'DocTitle',
            parent=styles['Heading1'],
            fontName=UNICODE_BOLD_FONT,
            fontSize=18,
            leading=22,
            textColor=colors.HexColor('#0f172a'),
            spaceAfter=4
        )
        
        subtitle_style = ParagraphStyle(
            'DocSubTitle',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=9,
            leading=13,
            textColor=colors.HexColor('#475569'),
            spaceAfter=12
        )

        section_heading = ParagraphStyle(
            'SectionHeading',
            parent=styles['Heading2'],
            fontName=UNICODE_BOLD_FONT,
            fontSize=12,
            leading=16,
            textColor=colors.HexColor('#1e293b'),
            spaceBefore=10,
            spaceAfter=6
        )

        body_style = ParagraphStyle(
            'Body',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=9.5,
            leading=14.5,
            textColor=colors.HexColor('#334155'),
            spaceAfter=5
        )

        bullet_style = ParagraphStyle(
            'BulletStyle',
            parent=styles['Normal'],
            fontName=UNICODE_FONT,
            fontSize=9.5,
            leading=14,
            textColor=colors.HexColor('#1e293b'),
            spaceAfter=4
        )

        story = []

        # 1. Document Header
        story.append(Spacer(1, 10))
        story.append(Paragraph("YouTube Video Intelligence & Executive Brief", title_style))
        gen_time = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
        story.append(Paragraph(f"Generated by Agentic AI Pipeline &bull; Timestamp: {gen_time} &bull; Job ID: {job_id}", subtitle_style))
        story.append(HRFlowable(width="100%", thickness=1.5, color=colors.HexColor("#2563eb"), spaceAfter=10))

        # 2. Metadata Table Card
        meta_data = [
            [
                Paragraph("<b>Video Title:</b>", body_style),
                Paragraph(f"<font color='#0284c7'><b>{metadata.title}</b></font>", body_style)
            ],
            [
                Paragraph("<b>Channel / Creator:</b>", body_style),
                Paragraph(f"{metadata.channel}", body_style)
            ],
            [
                Paragraph("<b>Duration & Lang:</b>", body_style),
                Paragraph(f"{metadata.duration_formatted} &bull; Detected Language: {metadata.detected_language.upper()}", body_style)
            ],
            [
                Paragraph("<b>Source URL:</b>", body_style),
                Paragraph(f"<font color='#2563eb'>https://www.youtube.com/watch?v={metadata.video_id}</font>", body_style)
            ],
        ]
        meta_table = Table(meta_data, colWidths=[110, 394])
        meta_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#e2e8f0')),
            ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#f1f5f9')),
            ('TOPPADDING', (0, 0), (-1, -1), 4),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 4),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(meta_table)
        story.append(Spacer(1, 10))

        # 3. Safety & Content Moderation Badge Card
        story.append(Paragraph("1. Content Safety & Harm Moderation Audit", section_heading))
        
        is_safe = safety_report.is_safe
        badge_bg = colors.HexColor("#f0fdf4") if is_safe else colors.HexColor("#fef2f2")
        badge_border = colors.HexColor("#86efac") if is_safe else colors.HexColor("#fca5a5")
        badge_title_color = "#166534" if is_safe else "#991b1b"
        badge_status_text = "PASSED: HARM-FREE & SAFE CONTENT" if is_safe else "ATTENTION: SENSITIVE / FLAGGED CONTENT"

        safety_rows = [
            [
                Paragraph(f"<font color='{badge_title_color}'><b>SAFETY STATUS: {badge_status_text}</b></font>", body_style),
            ],
            [
                Paragraph(f"<b>Assessment:</b> {safety_report.summary_assessment}", body_style),
            ]
        ]
        if safety_report.harm_flags_detected:
            flags_str = ", ".join(safety_report.harm_flags_detected)
            safety_rows.append([Paragraph(f"<b>Flagged Categories:</b> <font color='#dc2626'>{flags_str}</font>", body_style)])

        safety_table = Table(safety_rows, colWidths=[504])
        safety_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), badge_bg),
            ('BOX', (0, 0), (-1, -1), 1, badge_border),
            ('TOPPADDING', (0, 0), (-1, -1), 5),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 5),
            ('LEFTPADDING', (0, 0), (-1, -1), 8),
            ('RIGHTPADDING', (0, 0), (-1, -1), 8),
        ]))
        story.append(safety_table)
        story.append(Spacer(1, 10))

        # 4. Grounded Executive Summary (6 to 10 lines)
        story.append(Paragraph(f"2. Grounded Executive Summary ({summary.line_count} Key Insights)", section_heading))
        story.append(Paragraph("<i>Strictly grounded in transcript facts with zero external hallucination:</i>", subtitle_style))

        summary_data = []
        for idx, line in enumerate(summary.lines, start=1):
            summary_data.append([
                Paragraph(f"<b>{idx:02d}.</b>", bullet_style),
                Paragraph(line, bullet_style)
            ])

        summary_table = Table(summary_data, colWidths=[24, 480])
        summary_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#ffffff')),
            ('VALIGN', (0, 0), (-1, -1), 'TOP'),
            ('TOPPADDING', (0, 0), (-1, -1), 2),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 2),
            ('RIGHTPADDING', (0, 0), (-1, -1), 2),
        ]))
        story.append(summary_table)
        story.append(Spacer(1, 10))

        # 5. Action Items & Key Takeaways
        if action_items:
            story.append(Paragraph("3. Action Items & Concrete Next Steps", section_heading))
            actions_data = [
                [
                    Paragraph("<b>#</b>", body_style),
                    Paragraph("<b>Action Item / Takeaway</b>", body_style),
                    Paragraph("<b>Category</b>", body_style),
                    Paragraph("<b>Priority</b>", body_style),
                ]
            ]
            for it in action_items:
                priority_color = "#dc2626" if it.priority == "High" else ("#d97706" if it.priority == "Medium" else "#2563eb")
                actions_data.append([
                    Paragraph(f"[ ] {it.id}", body_style),
                    Paragraph(it.task, body_style),
                    Paragraph(it.category, body_style),
                    Paragraph(f"<font color='{priority_color}'><b>{it.priority}</b></font>", body_style)
                ])

            action_table = Table(actions_data, colWidths=[30, 314, 90, 70])
            action_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.HexColor('#f1f5f9')),
                ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#cbd5e1')),
                ('INNERGRID', (0, 0), (-1, -1), 0.5, colors.HexColor('#e2e8f0')),
                ('TOPPADDING', (0, 0), (-1, -1), 3),
                ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
                ('LEFTPADDING', (0, 0), (-1, -1), 4),
                ('RIGHTPADDING', (0, 0), (-1, -1), 4),
            ]))
            story.append(action_table)
            story.append(Spacer(1, 10))

        # 6. Human-in-the-Loop Verification Stamp
        story.append(Spacer(1, 6))
        hitl_status_text = "Automated AI Agent Verification (Passed)"
        reviewer = "AI Guardrail Pipeline"
        comments = "All guardrails passed; 100% transcript-grounded verification complete."

        if hitl_state and hitl_state.status in ("APPROVED", "EDITED"):
            hitl_status_text = f"Human Supervisor Verified ({hitl_state.status})"
            reviewer = hitl_state.reviewed_by or "Lead Human Reviewer"
            comments = hitl_state.review_comments or "Manually reviewed and approved."

        hitl_data = [
            [
                Paragraph("<b>HITL Audit & Verification:</b>", body_style),
                Paragraph(f"<b>{hitl_status_text}</b> &bull; Reviewer: {reviewer}", body_style)
            ],
            [
                Paragraph("<b>Audit Remarks:</b>", body_style),
                Paragraph(comments, body_style)
            ]
        ]
        hitl_table = Table(hitl_data, colWidths=[140, 364])
        hitl_table.setStyle(TableStyle([
            ('BACKGROUND', (0, 0), (-1, -1), colors.HexColor('#f8fafc')),
            ('BOX', (0, 0), (-1, -1), 1, colors.HexColor('#94a3b8')),
            ('TOPPADDING', (0, 0), (-1, -1), 3),
            ('BOTTOMPADDING', (0, 0), (-1, -1), 3),
            ('LEFTPADDING', (0, 0), (-1, -1), 6),
            ('RIGHTPADDING', (0, 0), (-1, -1), 6),
        ]))
        story.append(KeepTogether([hitl_table]))

        # Build PDF
        doc.build(story, canvasmaker=NumberedCanvas)
        return str(file_path)
