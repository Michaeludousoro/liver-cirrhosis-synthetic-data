"""Verify/render targeted PDFs and assemble the editable review handoff."""
from pathlib import Path
import shutil
import json
import re
import zipfile
import pypdfium2 as pdfium
from pypdf import PdfReader
from PIL import Image, ImageDraw

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
PAPER = HERE/'paper'
OUT = ROOT/'output/pdf'
OUT.mkdir(parents=True,exist_ok=True)
results = {}
for name in ('main','clean'):
    pdf = PAPER/f'{name}.pdf'
    reader = PdfReader(pdf)
    text = '\n'.join(p.extract_text() for p in reader.pages)
    assert '??' not in text
    assert 'Theory of Augmentation for Improved Prediction Accuracy' in text
    assert all(s in text for s in ['0.8227','0.8081','0.8184','0.8377','0.8088','0.7803'])
    assert '42–46' in text or '42--46' in text or '42–' in text
    results[name] = {'pages':len(reader.pages),'unresolved_references':False}
    out = HERE/'qa'/name
    out.mkdir(parents=True,exist_ok=True)
    doc = pdfium.PdfDocument(str(pdf))
    thumbs = []
    for i in range(len(doc)):
        im = doc[i].render(scale=1.4).to_pil().convert('RGB')
        im.save(out/f'page-{i+1:02d}.png')
        im.thumbnail((460,620))
        thumbs.append(im)
    for start in range(0,len(thumbs),6):
        sheet = Image.new('RGB',(1440,1320),'#d8dde3')
        draw = ImageDraw.Draw(sheet)
        for k, im in enumerate(thumbs[start:start+6]):
            x,y=(k%3)*480+10,(k//3)*660+30
            sheet.paste(im,(x,y))
            draw.text((x,y-20),f'{name}: page {start+k+1}',fill='black')
        sheet.save(out/f'contact-{start//6+1}.png')
    shutil.copy2(pdf,OUT/('Liver_Cirrhosis_Targeted_Blue_Review.pdf' if name=='main' else 'Liver_Cirrhosis_Targeted_Clean_Review.pdf'))

shutil.copy2(HERE/'REVIEWER_CORRECTION_RECORD.md',OUT/'Liver_Cirrhosis_Targeted_Correction_Record.md')
(HERE/'qa-results.json').write_text(json.dumps(results,indent=2))
ledger = json.loads((HERE/'change-ledger.json').read_text())
lines = ['# Detailed replacement ledger','', 'Each entry records the original text, replacement, and reason. Layout-only ACM conversion is excluded. Abstract shortening, the retained title and five keywords are documented in the main correction record.','']
for i,entry in enumerate(ledger,1):
    lines += [f'## {i}. {entry["category"]}', '',entry['reason'],'','Original:','', '```tex',entry['original'],'```','','Replacement:','','```tex',entry['replacement'] or '[Removed]','```','']
(HERE/'DETAILED_CHANGE_LEDGER.md').write_text('\n'.join(lines))
with zipfile.ZipFile(OUT/'Liver_Cirrhosis_Targeted_Review_Package.zip','w',zipfile.ZIP_DEFLATED) as z:
    for file in PAPER.rglob('*'):
        if file.is_file() and file.suffix in {'.tex','.bib','.cls','.bst','.png','.jpg'}:
            z.write(file,'paper/'+str(file.relative_to(PAPER)))
    for name in ['REVIEWER_CORRECTION_RECORD.md','DETAILED_CHANGE_LEDGER.md','change-ledger.json','word-counts.json','qa-results.json']:
        z.write(HERE/name,name)
    for name in ['cv_results.csv','cv_folds.csv','reproduction_checks.json','artifact_manifest.json']:
        z.write(HERE.parent/'2026-09-17/validation'/name,'validation-summary/'+name)
    for name in ['Liver_Cirrhosis_Targeted_Blue_Review.pdf','Liver_Cirrhosis_Targeted_Clean_Review.pdf']:
        z.write(OUT/name,name)
print(json.dumps(results,indent=2))
