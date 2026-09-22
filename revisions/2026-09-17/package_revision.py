"""Package final sources, public-data artifacts and completed rerun provenance."""
from pathlib import Path
import shutil
import subprocess
import zipfile

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OUT = ROOT/'output/pdf'
OUT.mkdir(exist_ok=True)
historical = HERE/'validation/historical_results'
historical.mkdir(exist_ok=True)
for name in ['model_performance','smote_results','scenario_e_six_generators']:
    (historical/f'{name}.csv').write_bytes(subprocess.check_output(
        ['git','show',f'HEAD:output/results/{name}.csv'],cwd=ROOT))
shutil.copy2(HERE/'paper/main.pdf',OUT/'Liver_Cirrhosis_ACM_Blue_Review.pdf')
shutil.copy2(HERE/'paper/clean.pdf',OUT/'Liver_Cirrhosis_ACM_Clean_Review.pdf')
files = [p for p in (HERE/'paper').rglob('*') if p.is_file() and p.suffix in {'.tex','.bib','.bst','.cls','.png'}
         and p.name != 'figS16_pipeline_flowchart.png']
files += [p for p in (HERE/'validation').rglob('*') if p.is_file()]
files += [HERE/'README.md',HERE/'requirements-validation.txt',HERE/'audit_and_validate.py',
          HERE/'verify_artifacts.py',HERE/'test_validation.py']
modules = ['__init__','data_loader','predictive_modeling','statistical_analysis',
           'fold_generation','synthetic_generator','seeding','consensus_voting','iqr_filter','fid_calculator']
files += [ROOT/f'src/{name}.py' for name in modules if (ROOT/f'src/{name}.py').exists()]
files += [ROOT/'data/raw/cirrhosis.csv']
files += [ROOT/f'output/data/{name}.csv' for name in ['train_real','filtered_gan','filtered_ctgan',
          'filtered_tvae','consensus_equalised','filtered_masked_vae','filtered_ctgan_proper']]
target = OUT/'Liver_Cirrhosis_ACM_Review_Package.zip'
with zipfile.ZipFile(target,'w',zipfile.ZIP_DEFLATED) as package:
    for path in files:
        package.write(path,str(path.relative_to(ROOT)))
    package.write(HERE/'validation/test_real_reconstructed.csv','output/data/test_real.csv')
    package.write(HERE/'README.md','START_HERE.md')
with zipfile.ZipFile(target) as package:
    assert package.testzip() is None
    print('Archive verified:',len(package.namelist()),'files;',target.stat().st_size,'bytes')
