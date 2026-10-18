"""Estilos compartidos para los informes PDF (reportlab)."""
from reportlab.lib import colors
from reportlab.lib.enums import TA_JUSTIFY
from reportlab.lib.pagesizes import letter
from reportlab.lib.styles import ParagraphStyle
from reportlab.lib.units import cm
from reportlab.pdfbase import pdfmetrics
from reportlab.pdfbase.ttfonts import TTFont
from reportlab.platypus import Image, Paragraph, SimpleDocTemplate, Spacer, Table, TableStyle

D = "/usr/share/fonts/truetype/dejavu/"
pdfmetrics.registerFont(TTFont("DV", D + "DejaVuSans.ttf"))
pdfmetrics.registerFont(TTFont("DV-B", D + "DejaVuSans-Bold.ttf"))
pdfmetrics.registerFont(TTFont("DV-I", D + "DejaVuSans-Oblique.ttf"))
pdfmetrics.registerFont(TTFont("DV-BI", D + "DejaVuSans-BoldOblique.ttf"))
from reportlab.pdfbase.pdfmetrics import registerFontFamily  # noqa: E402

registerFontFamily("DV", normal="DV", bold="DV-B", italic="DV-I", boldItalic="DV-BI")

AZUL = colors.HexColor("#1f4e8c")
TINTA = colors.HexColor("#1b1b1b")
GRIS = colors.HexColor("#5a5955")
LINEA = colors.HexColor("#d9d8d3")
FONDO = colors.HexColor("#f3f5f9")

ST = {
    "titulo": ParagraphStyle("t", fontName="DV-B", fontSize=15, leading=19, textColor=AZUL, spaceAfter=4),
    "sub": ParagraphStyle("s", fontName="DV", fontSize=9, leading=12, textColor=GRIS, spaceAfter=8),
    "h1": ParagraphStyle("h1", fontName="DV-B", fontSize=11, leading=14, textColor=AZUL, spaceBefore=8, spaceAfter=3, keepWithNext=1),
    "h2": ParagraphStyle("h2", fontName="DV-B", fontSize=9.5, leading=12, textColor=TINTA, spaceBefore=5, spaceAfter=2, keepWithNext=1),
    "p": ParagraphStyle("p", fontName="DV", fontSize=8.6, leading=11.6, textColor=TINTA, alignment=TA_JUSTIFY, spaceAfter=4),
    "li": ParagraphStyle("li", fontName="DV", fontSize=8.6, leading=11.4, textColor=TINTA, leftIndent=10, bulletIndent=2, spaceAfter=1.5),
    "cap": ParagraphStyle("c", fontName="DV-I", fontSize=7.6, leading=9.5, textColor=GRIS, spaceAfter=6),
    "td": ParagraphStyle("td", fontName="DV", fontSize=7.6, leading=9.4, textColor=TINTA),
    "th": ParagraphStyle("th", fontName="DV-B", fontSize=7.6, leading=9.4, textColor=colors.white),
    "ref": ParagraphStyle("r", fontName="DV", fontSize=7.4, leading=9.4, textColor=TINTA, leftIndent=10, firstLineIndent=-10, spaceAfter=1.5),
}


def P(t, s="p"):
    return Paragraph(t, ST[s])


def bullets(items):
    return [Paragraph(i, ST["li"], bulletText="•") for i in items]


def tabla(filas, anchos, cab=True, zebra=True):
    data = [[c if not isinstance(c, str) else Paragraph(c, ST["th"] if (cab and i == 0) else ST["td"]) for c in f]
            for i, f in enumerate(filas)]
    t = Table(data, colWidths=[a * cm for a in anchos], repeatRows=1 if cab else 0)
    sty = [("VALIGN", (0, 0), (-1, -1), "TOP"), ("LINEBELOW", (0, 0), (-1, -1), 0.4, LINEA),
           ("TOPPADDING", (0, 0), (-1, -1), 2.5), ("BOTTOMPADDING", (0, 0), (-1, -1), 2.5),
           ("LEFTPADDING", (0, 0), (-1, -1), 4), ("RIGHTPADDING", (0, 0), (-1, -1), 4)]
    if cab:
        sty.append(("BACKGROUND", (0, 0), (-1, 0), AZUL))
    if zebra:
        for r in range(1 if cab else 0, len(filas)):
            if r % 2 == 0:
                sty.append(("BACKGROUND", (0, r), (-1, r), FONDO))
    t.setStyle(TableStyle(sty))
    return t


def figura(path, ancho_cm, cap=None):
    from PIL import Image as PI

    w, h = PI.open(path).size
    img = Image(str(path), width=ancho_cm * cm, height=ancho_cm * cm * h / w)
    return [img, P(cap, "cap")] if cap else [img]


def construir(path, story, titulo_corto):
    def pie(c, doc):
        c.saveState()
        c.setFont("DV", 7)
        c.setFillColor(GRIS)
        c.drawString(2 * cm, 1.2 * cm, titulo_corto)
        c.drawRightString(letter[0] - 2 * cm, 1.2 * cm, f"Página {doc.page}")
        c.restoreState()

    doc = SimpleDocTemplate(str(path), pagesize=letter, leftMargin=2 * cm, rightMargin=2 * cm, topMargin=1.7 * cm,
                            bottomMargin=1.8 * cm, title=titulo_corto, author="Grupo 7")
    doc.build(story, onFirstPage=pie, onLaterPages=pie)


__all__ = ["P", "bullets", "tabla", "figura", "construir", "Spacer", "cm", "ST"]
