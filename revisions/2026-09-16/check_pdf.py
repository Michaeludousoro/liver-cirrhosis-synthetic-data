"""Render manuscript pages and compact contact sheets for visual QA."""
from pathlib import Path
import pypdfium2 as pdfium
from PIL import Image, ImageDraw
from pypdf import PdfReader

root = Path(__file__).resolve().parents[2]
pdf = root / 'revisions/2026-09-16/paper/main.pdf'
out = root / 'tmp/pdfs'
doc = pdfium.PdfDocument(str(pdf))
pages = []
for i in range(len(doc)):
    page = doc[i].render(scale=1.5).to_pil().convert('RGB')
    page.save(out / f'final-{i+1:02d}.png')
    thumb = page.copy()
    thumb.thumbnail((680, 880))
    pages.append(thumb)
for start in range(0, len(pages), 4):
    sheet = Image.new('RGB', (1400, 1840), '#d8dde3')
    draw = ImageDraw.Draw(sheet)
    for k, page in enumerate(pages[start:start+4]):
        x, y = (k % 2) * 700 + 10, (k // 2) * 920 + 28
        sheet.paste(page, (x, y))
        draw.text((x, y - 20), f'Page {start+k+1}', fill='black')
    sheet.save(out / f'contact-{start//4+1}.png')
reader = PdfReader(pdf)
print('Pages:', len(reader.pages))
for i, page in enumerate(reader.pages):
    text = page.extract_text()
    print(i+1, len(text), 'PLACEHOLDER' if '??' in text else '')
