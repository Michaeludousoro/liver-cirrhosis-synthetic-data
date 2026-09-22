"""Render both ACM PDFs for page-by-page quality review."""
from pathlib import Path
import sys
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

HERE = Path(__file__).resolve().parent
for name in (sys.argv[1:] or ['main', 'clean']):
    pdf = HERE / 'paper' / f'{name}.pdf'
    out = HERE / 'qa' / name
    out.mkdir(parents=True, exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    thumbs = []
    for i in range(len(doc)):
        page = doc[i].render(scale=1.3).to_pil().convert('RGB')
        page.save(out / f'page-{i+1:02d}.png')
        page.thumbnail((460, 620))
        thumbs.append(page)
    for start in range(0, len(thumbs), 6):
        sheet = Image.new('RGB', (1440, 1320), '#d8dde3')
        draw = ImageDraw.Draw(sheet)
        for k, page in enumerate(thumbs[start:start+6]):
            x, y = (k % 3)*480+10, (k//3)*660+30
            sheet.paste(page, (x, y))
            draw.text((x,y-20), f'{name}: page {start+k+1}', fill='black')
        sheet.save(out / f'contact-{start//6+1}.png')
    reader = PdfReader(pdf)
    texts = [p.extract_text() for p in reader.pages]
    assert not any('??' in t for t in texts), 'Unresolved reference marker'
    print(name, len(texts), 'pages; no unresolved reference markers')
