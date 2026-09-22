"""Mechanical ACM conversion and paragraph-level blue comparison to supplied ZIP.

The canonical manuscript is content.tex. main.tex shows revisions; clean.tex
uses exactly the same content with marking disabled. Figures remain true-colour.
"""
from pathlib import Path
import re

HERE = Path(__file__).resolve().parent
PAPER = HERE / 'paper'
SOURCE = HERE.parent / '2026-09-16/paper/main.tex'
BASELINE = Path('/tmp/ieee-article-preview.xJzjKz/IEEE_Synthetic Data_latex/paper/main.tex')


def bibliography(text):
    entries = re.findall(r'\\bibitem\{([^}]+)\}\s*(.*?)(?=\\bibitem|\\end\{thebibliography\})', text, re.S)
    special = {
        'fleming1991': ('book', {'author':'T. R. Fleming and D. P. Harrington', 'title':'Counting Processes and Survival Analysis', 'publisher':'Wiley', 'address':'New York, NY', 'year':'1991'}),
        'stenger2024': ('article', {'author':'Michael Stenger and André Bauer and Thomas Prantl and Robert Leppich and Nathaniel Hudson and Kyle Chard and Ian Foster and Samuel Kounev', 'title':'Thinking in Categories: A Survey on Assessing the Quality for Time Series Synthesis', 'journal':'Journal of Data and Information Quality', 'year':'2024', 'doi':'10.1145/3666006'}),
        'mamun2026': ('article', {'author':'Abdullah Mamun and Lawrence D. Devoe and Mark I. Evans and David W. Britt and Judith Klein-Seetharaman and Hassan Ghasemzadeh', 'title':'Use of What-if Scenarios to Help Explain Artificial Intelligence Models for Neonatal Health', 'journal':'ACM Transactions on Computing for Healthcare', 'year':'2026', 'doi':'10.1145/3814951'}),
        'nistfences': ('misc', {'author':'{NIST/SEMATECH}', 'title':'What are outliers in the data?', 'howpublished':'e-Handbook of Statistical Methods, Section 7.1.6', 'url':'https://www.itl.nist.gov/div898/handbook/prc/section1/prc16.htm', 'urldate':'2026-09-16'}),
        'pedregosa2011': ('article', {'author':'F. Pedregosa and others', 'title':'Scikit-learn: Machine Learning in Python', 'journal':'Journal of Machine Learning Research', 'volume':'12', 'pages':'2825--2830', 'year':'2011', 'url':'https://jmlr.org/papers/v12/pedregosa11a.html'}),
    }
    out = []
    for key, entry in entries:
        if key in special:
            kind, fields = special[key]
            if key == 'stenger2024':
                fields.update({'volume':'16','number':'2','pages':'1--32'})
            if key == 'mamun2026':
                fields['note'] = 'Accepted manuscript, online 26 May 2026'
        else:
            authors, tail = entry.split('``', 1)
            title, tail = tail.split(",''", 1)
            authors = authors.strip().rstrip(',').replace('~', ' ')
            authors = authors.replace(r'\textit{et al.}', 'and others')
            authors = authors.replace(', and ', ', ').replace(', ', ' and ')
            fields = {'author':authors, 'title':title, 'year':re.search(r'(?:19|20)\d{2}', tail).group()}
            venue = re.search(r'\\textit\{([^}]+)\}', tail)
            kind = 'article'
            if ' in ' in tail:
                kind = 'inproceedings'
            if venue:
                fields['booktitle' if kind == 'inproceedings' else 'journal'] = venue.group(1)
            else:
                kind = 'misc'
                fields['howpublished'] = 'UCI Machine Learning Repository'
            for pattern, field in [(r'vol\.~([^,]+)', 'volume'), (r'no\.~([^,]+)', 'number'), (r'pp?\.~([^,]+)', 'pages'), (r'doi:~([^\s]+)', 'doi'), (r'\\url\{([^}]+)\}', 'url')]:
                match = re.search(pattern, tail)
                if match:
                    fields[field] = match.group(1).rstrip('.')
        out.append('@' + kind + '{' + key + ',\n' + ',\n'.join('  '+k+' = {'+('{' + v + '}' if k == 'title' else v)+'}' for k,v in fields.items()) + '\n}\n')
    (PAPER / 'references.bib').write_text('\n'.join(out))


def prepare():
    text = SOURCE.read_text()
    baseline = BASELINE.read_text()
    bibliography(text)
    text = re.sub(r'\\begin\{thebibliography\}.*?\\end\{thebibliography\}',
                  lambda m: '\\begingroup\n\\revisioncolor\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{references}\n\\endgroup', text, flags=re.S)
    text = text[text.index(r'\section{Introduction}'):]
    text = text.replace(r'\end{document}', '')
    text = text.replace(r'\resizebox{\columnwidth}{!}{', r'\fitwidth{')

    # Correct reporting now; insert only completed fold-contained estimates.
    start = text.index('Five-fold stratified validation partitions')
    end = text.index('\n\n', start)
    text = text[:start] + ('Five-fold stratified validation partitions the 193 real training patients; each validation fold contains only real patients. The revised implementation rejects precomputed synthetic pools and requires preprocessing, generator fitting, IQR bounds, equalisation, and consensus selection to use only the corresponding training fold. SMOTE is likewise fitted within each fold. Generator settings and the consensus tolerance of 5.0 are fixed; this is fold-contained validation, not nested hyperparameter selection. Any future tuning must use an inner training-only split. The earlier cGAN and consensus cross-validation estimates are withdrawn because their generators had seen validation-fold patients. Replacement generative estimates are not reported until retraining is complete. The 83-patient test cohort is excluded from fitting. Artifact-based test comparisons and baseline/SMOTE cross-validation were rerun separately; the accompanying audit records software versions and the reconstructed fixed split.') + text[end:]
    start = text.index('Table~\\ref{tab:cv} reports exploratory')
    end = text.index(r'\subsection{Bootstrap Confidence', start)
    import csv
    rows = list(csv.DictReader((HERE / 'validation/cv_results.csv').open()))
    values = {(r['Scenario'], r['Classifier']):r for r in rows}
    table = ['The previous generative cross-validation results are withdrawn because generator fitting preceded fold partitioning. Table~\\ref{tab:cv} reports rerun baseline and within-fold SMOTE estimates only. The corrected generator interface enforces fold-specific fitting, but no replacement generative estimates are claimed here. Mean and standard deviation summarise the five folds; they are not a confidence interval or an independent test of augmentation benefit.', '', r'\begin{table}[tb]', r'\centering', r'\caption{Rerun Five-Fold AUC: Mean $\pm$ Fold Standard Deviation; Follow-up Duration Excluded}', r'\label{tab:cv}', r'\begin{tabular}{lccc}', r'\toprule', r'Scenario & RF & GB & LR \\', r'\midrule']
    complete_cv = len(rows) == 12
    for scenario in (['A: Baseline','B: Real + cGAN','C: Real + Consensus','D: Real + SMOTE'] if complete_cv else ['A: Baseline','D: Real + SMOTE']):
        cells = [f"${float(values[(scenario,c)]['Mean AUC']):.4f} \\pm {float(values[(scenario,c)]['Std AUC']):.4f}$" for c in ['Random Forest','Gradient Boosting','Logistic Regression']]
        table.append(scenario + ' & ' + ' & '.join(cells) + r' \\')
    if not complete_cv:
        table += [r'B: Real + cGAN & \multicolumn{3}{c}{Withdrawn; fold-specific rerun required} \\', r'C: Real + Consensus & \multicolumn{3}{c}{Withdrawn; fold-specific rerun required} \\']
    else:
        table[0] = ('Table~\\ref{tab:cv} replaces the earlier generative cross-validation estimates with a completed fold-contained rerun. Each of the five folds retrains GAN, cGAN, and VAE using only its 154 or 155 real training patients, generates 500 records per model, and recomputes IQR filtering, equalisation, and consensus. Validation uses the remaining 38 or 39 real patients. Both generative augmentation strategies have lower mean AUC than the real-only baseline for all three classifiers. For consensus, LR mean AUC is 0.7803 rather than the earlier, fold-leaked estimate of 0.8349. SMOTE yields a small descriptive RF increase from 0.8499 to 0.8559. Means and standard deviations summarise five correlated folds; these comparisons are not significance tests or confidence intervals.')
    table += [r'\bottomrule', r'\end{tabular}', r'\end{table}', '']
    text = text[:start] + '\n'.join(table) + '\n' + text[end:]
    replacements = {
      'Existing cross-validation results reuse fitted synthetic pools and require fold-specific regeneration before confirmatory interpretation.':'Earlier generative cross-validation estimates have been withdrawn; the revised implementation requires fold-specific generation before replacement estimates can be reported.',
      'Cross-validation provides exploratory context but cannot independently confirm the generative comparisons because the synthetic pools were fitted before fold partitioning. SMOTE is fitted within each fold and yields a small descriptive increase in RF mean AUC. A fully nested comparison is required to evaluate whether either generative augmentation or interpolation provides a reproducible benefit.':'Rerun baseline and within-fold SMOTE cross-validation provide descriptive context. The small RF mean-AUC increase with SMOTE is not a formal paired test of improvement. Earlier generative cross-validation estimates are withdrawn; fold-specific retraining is required before comparing generative augmentation on this basis.',
      'The cross-validation implementation reuses synthetic pools trained on all 193 training patients. Fully nested generation and preprocessing are required to eliminate validation-fold information from the augmented training data. The reported mean plus or minus 1.96 fold standard deviations is not a confidence interval for the mean; fold means and standard deviations are retained only as exploratory summaries.':'The original generative cross-validation reused pools fitted to all 193 training patients and is withdrawn. The corrected code requires training-fold-only preprocessing, generation, filtering, and selection, and rejects precomputed pools. Until the corrected generative run is complete, this paper reports only rerun baseline and SMOTE cross-validation. Fold standard deviations measure split variability and are not confidence intervals; correlated folds and a single split limit inference.',
      'and generative cross-validation reuses pools fitted before fold partitioning.':'and independent generative cross-validation remains incomplete after withdrawal of estimates affected by fold leakage.',
      'Future work should regenerate synthetic pools within each validation fold, evaluate paired performance differences,':'The immediate validation priority is to complete the corrected fold-specific generator rerun, evaluate paired performance differences,',
      'classic augmentation methods have its own limitations.':'classic augmentation methods have their own limitations.',
      'In contrast, deep generative models handle these limitations better, since they learn those relationships directly from the data, and they are now one of the most common ways to generate synthetic clinical tabular data':'Deep generative models can learn nonlinear relationships directly from data, although this flexibility does not guarantee better utility or fidelity',
      'While VAE uses a different strategy; instead of adversarial training,':'A VAE uses a different strategy: instead of adversarial training,',
      'However good distributional similarity':'However, good distributional similarity',
      "The main question about following the path from distributional fidelity to predictive utility has received much less attention, especially for small tabular clinical datasets used in IoMT environments. \\cite{kababji2023}, \\cite{espinosa2023}, and that is the main motivation of this study to test predictive utility on unseen real patients, not just similarity metrics.":'These findings motivate testing predictive utility on unseen real patients alongside distributional similarity, particularly in small clinical cohorts \\cite{kababji2023,espinosa2023}.',
      'on the leak-free feature set':'using predictors that exclude follow-up duration',
      'Leak-Free 17-Feature Input':'17 Baseline Predictors',
      'Every quantity plotted in S6 to S10 also appears numerically in Tables~\\ref{tab:performance} and~\\ref{tab:cv}.':'The supplementary plots are historical artifact-based analyses; they must not be used as evidence for the withdrawn generative cross-validation estimates.',
    }
    for old, new in replacements.items():
        text = text.replace(old, new)
    if complete_cv:
        start = text.index('Five-fold stratified validation partitions')
        end = text.index('\n\n', start)
        text = text[:start] + ('Five-fold stratified validation partitions the 193 real training patients before fitting any generator. Each training fold independently fits min-max scaling, GAN (500 epochs), cGAN (300 epochs), and VAE (300 epochs); each model generates 500 records. IQR bounds use only the real training fold. The filtered VAE pool is downsampled to the smaller adversarial-pool size, and consensus is recomputed at tolerance 5.0. Classifier scaling is fitted only to the corresponding augmented training fold; SMOTE is fitted within the fold. Validation uses only real patients, and the 83-patient test cohort is excluded throughout. The implementation rejects precomputed pools and stops rather than substituting another method if consensus fails. Generator seeds are 42--46 for folds 1--5; the split and classifier seeds are 42. Settings are fixed from the existing protocol, with no inner hyperparameter search. This is retrospective fold-contained internal validation, not nested model selection or external validation. Python 3.9.6, TensorFlow 2.15.0, NumPy 1.26.4, and scikit-learn 1.5.2 were used for the corrected rerun. Per-fold membership, scaling bounds, generated pools, and AUCs are retained in the accompanying artifact.') + text[end:]
        text = text.replace('Earlier generative cross-validation estimates have been withdrawn; the revised implementation requires fold-specific generation before replacement estimates can be reported.', 'Generative cross-validation was rerun with independent fitting, filtering, and consensus construction inside each training fold.')
        text = text.replace('Rerun baseline and within-fold SMOTE cross-validation provide descriptive context. The small RF mean-AUC increase with SMOTE is not a formal paired test of improvement. Earlier generative cross-validation estimates are withdrawn; fold-specific retraining is required before comparing generative augmentation on this basis.', 'The corrected cross-validation results do not support an augmentation benefit from cGAN or consensus: their mean AUCs are below the real-only baseline for all three classifiers. The small RF mean-AUC increase with within-fold SMOTE is descriptive, not a formal paired test of improvement. These results reinforce the distinction between synthetic-data resemblance and useful augmentation.')
        text = text.replace('The original generative cross-validation reused pools fitted to all 193 training patients and is withdrawn. The corrected code requires training-fold-only preprocessing, generation, filtering, and selection, and rejects precomputed pools. Until the corrected generative run is complete, this paper reports only rerun baseline and SMOTE cross-validation. Fold standard deviations measure split variability and are not confidence intervals; correlated folds and a single split limit inference.', 'The original generative cross-validation reused pools fitted to all 193 training patients; its estimates are superseded by the fold-contained rerun. The revised code and recorded fold memberships exclude validation patients from preprocessing, generator fitting, and filtering. Nevertheless, this is a retrospective internal evaluation of an existing protocol. Hyperparameters were not selected in a nested inner loop, only one generator seed was used per fold, and fold standard deviations are not confidence intervals. Repeated-seed evaluation and external cohorts remain necessary.')
        text = text.replace('and independent generative cross-validation remains incomplete after withdrawal of estimates affected by fold leakage.', 'and the corrected cross-validation is retrospective and internal rather than an external evaluation.')
        text = text.replace('The immediate validation priority is to complete the corrected fold-specific generator rerun, evaluate paired performance differences,', 'Future work should repeat fold-specific generation across multiple seeds, evaluate paired performance differences,')
        text = text.replace('The results demonstrate that synthetic records can retain useful predictive structure without consistently improving models trained on available real data.', 'The results demonstrate that synthetic records can retain predictive structure without consistently improving real-data models. Corrected fold-contained validation found lower mean AUC with cGAN and consensus augmentation for all three classifiers.')
    text = text.replace('sigmoid output of dimension 18', 'sigmoid output of dimension 19 (18 generated variables plus the outcome label)')
    text = text.replace('Both generator and discriminator are conditioned on the outcome label $\\mathbf{y}$, represented as a two-dimensional one-hot vector,', 'Both networks condition on the binary outcome label $\\mathbf{y}$, encoded as a two-element one-hot vector,')
    text = text.replace('Each feature is standardised as', 'Consensus distances include all 19 generated columns, including follow-up duration and the binary outcome label. Each column is standardised as')
    text = text.replace('Seeds are fixed at 42.', 'The fixed split and classifiers use seed 42; the corrected generator rerun uses seeds 42--46 across its five folds.')
    text = text.replace("Cohen's $d$ ranks the VAE as the best-matched method on every feature even though it is the worst on every distributional-shape measure, which is a further instance of the paper's central point that a single fidelity metric can be satisfied by a model that is wrong in every other respect.", "the VAE has small absolute standardised mean differences despite substantial distributional-shape differences. It is not the closest method on every individual mean-difference measure. These results reinforce the need for multiple quality criteria.")
    descriptions = {
        'fig13_training_raincloud.png':'Training-only biomarker distributions stratified by recorded status; points, densities, and box plots show the small-sample spread.',
        'fig14_study_overview.png':'The train and test split precedes generation; fitted pools undergo filtering and consensus before distinct fidelity, disclosure, and prediction evaluations.',
        'fig06_roc_curves.png':'Three panels compare held-out receiver operating characteristic curves for Random Forest, Gradient Boosting, and logistic regression across training scenarios.',
        'fig07_feature_importance.png':'Horizontal bars rank baseline Random Forest and Gradient Boosting predictor importance, with bilirubin ranked first in both.',
        'fig08_power_analysis.png':'A sensitivity analysis relates discordant-pair counts to McNemar p-values and assumed effect sizes to required test sample sizes.',
        'fig11_fid_vs_tstr.png':'Six generator methods are plotted by continuous-feature Frechet distance and mean synthetic-only test AUC; their rankings differ.',
    }
    for filename, description in descriptions.items():
        text = re.sub(r'(\\includegraphics[^\n]*\{'+re.escape(filename)+r'\})', lambda m: m.group(1)+'\n\\Description{'+description+'}', text)
    # Mark changed paragraphs/environments relative to the user-supplied file.
    def norm(s):
        s = re.sub(r'(?m)^\s*%.*$', '', s)
        return re.sub(r'\s+', ' ', s).strip()
    original = norm(baseline)
    output = []
    lines = text.splitlines()
    i = 0
    environments = ['table','table*','figure','figure*','equation','equation*','align','align*','enumerate','itemize']
    changed = 0
    while i < len(lines):
        line = lines[i]
        match = re.match(r'\s*\\begin\{([^}]+)\}', line)
        if match and match.group(1) in environments:
            env = match.group(1)
            j = i + 1
            while j < len(lines) and r'\end{'+env+'}' not in lines[j]:
                j += 1
            block = '\n'.join(lines[i:j+1])
            if norm(block) not in original:
                # Place colour inside floats so LaTeX's float reset cannot erase it.
                block = block.replace(line, line+'\n\\revisioncolor', 1)
                block = block.replace(r'\caption{', r'\caption{\revisioncolor ')
                if env not in ['table','table*','figure','figure*']:
                    block = '\\begingroup\n' + block + '\n\\endgroup'
                changed += 1
            output.append(block)
            i = j + 1
            continue
        if line.strip() and not line.lstrip().startswith(('%','\\')) and norm(line) not in original:
            output.append(r'\rev{'+line+'}')
            changed += 1
        else:
            output.append(line)
        i += 1
    (PAPER/'content.tex').write_text('\n'.join(output)+'\n')
    abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}', SOURCE.read_text(), re.S).group(1).strip()
    abstract = abstract.replace('fully nested validation', 'fold-contained generative validation')
    if complete_cv:
        abstract = abstract.replace('Larger external cohorts and fold-contained generative validation are needed to establish generalisable augmentation benefits and assess disclosure risk.', 'Fold-contained cross-validation showed no consistent augmentation gain. External cohorts remain necessary to assess generalisability and disclosure risk.')
    header = r'''\documentclass[manuscript,review,screen,nonacm]{acmart}
\usepackage{multirow}
\usepackage{placeins}
\usepackage{float}
\usepackage{enumitem}
\graphicspath{{figures/}}
\acmJournal{JDIQ}
\setcopyright{none}
\settopmatter{printacmref=false}
\hypersetup{hidelinks}
\pdfstringdefDisableCommands{\def\rev#1{#1}}
\acmDOI{}
\acmISBN{}
\newif\ifshowrevisions
\ifdefined\cleanversion\showrevisionsfalse\else\showrevisionstrue\fi
\definecolor{revisionblue}{RGB}{0,55,170}
\DeclareRobustCommand{\rev}[1]{\ifshowrevisions{\color{revisionblue}#1}\else#1\fi}
\newcommand{\revisioncolor}{\ifshowrevisions\color{revisionblue}\fi}
\newcommand{\fitwidth}[1]{\begingroup\sbox0{#1}\ifdim\wd0>\linewidth\resizebox{\linewidth}{!}{\usebox0}\else\usebox0\fi\endgroup}
\begin{document}
\title[Synthetic Liver Cirrhosis Data: Utility and Quality]{\rev{Evaluating Synthetic Liver Cirrhosis Data: Predictive Utility, Distributional Fidelity, and Consensus-Based Quality Filtering}}
\author{Michael Udousoro}
\affiliation{\institution{University of Roehampton}\department{School of Computing, Engineering and the Built Environment}\city{London}\country{United Kingdom}}
\author{Mohammad Farhan Khan}
\affiliation{\institution{University of Roehampton}\department{School of Computing, Engineering and the Built Environment}\city{London}\country{United Kingdom}}
\author{Fakhreldin Saeed}
\affiliation{\institution{University of Roehampton}\department{School of Computing, Engineering and the Built Environment}\city{London}\country{United Kingdom}}
\author{M. Mursaleen}
\authornote{Corresponding author.}
\email{mursaleenm@gmail.com}
\affiliation{\institution{China Medical University Hospital, China Medical University}\department{Department of Medical Research}\city{Taichung}\country{Taiwan}}
\renewcommand{\shortauthors}{Udousoro et al.}
\begin{abstract}
'''+r'\rev{'+abstract+'}\n'+r'''\end{abstract}
\ccsdesc[500]{Computing methodologies~Machine learning}
\ccsdesc[300]{Applied computing~Health informatics}
\ccsdesc[300]{Information systems~Data cleaning}
\keywords{Synthetic clinical data, predictive utility, data quality, liver cirrhosis, generative models, consensus filtering, disclosure risk}
\maketitle
\ifshowrevisions
\noindent\rev{\textbf{Reviewer copy, 17 September 2026.} Blue text identifies revised or added paragraphs, tables, captions, and reformatted references relative to the supplied IEEE manuscript. Unchanged prose remains black. Figures retain their original colours. Historical fold-leaked validation estimates have been replaced by a completed fold-contained rerun.}
\medskip
\fi
\input{content}
\end{document}
'''
    (PAPER/'main.tex').write_text(header)
    (PAPER/'clean.tex').write_text('\\def\\cleanversion{1}\n\\input{main.tex}\n')
    prepare_supplement(header)
    print(f'Prepared ACM sources; {changed} changed prose/environment blocks marked blue.')


def prepare_supplement(header):
    source = (HERE.parent/'2026-09-16/paper/supplementary.tex').read_text()
    original = (BASELINE.parent/'supplementary.tex').read_text()
    text = source[source.index(r'\section{Scope of This Document}'):source.index(r'\begin{thebibliography}')]
    # Rewrite overclaims and align supporting methods with the audited code.
    paragraphs = {
        'Nothing in this document changes any conclusion': 'The figures and distributional tables describe the saved full-training artifacts, not newly trained fold-specific pools. The main manuscript now reports a separate corrected five-fold generator rerun, with per-fold provenance and outputs included in the revision package. Historical supplementary plots must not be mistaken for that rerun.',
        'All features are scaled': 'For the full-training generative analysis, min-max scaling is fitted to the 193 real training patients and inverted after generation. The synthetic table has 19 columns: 18 generated variables and Status. Classifiers exclude N\\_Days and use 17 baseline predictors, with separate training-fitted standardisation. In the corrected cross-validation, every scaler, generator, IQR bound, and consensus pool is fitted or constructed anew inside the corresponding training fold.',
        'The project uses fixed seeds': 'The full-training artifact analyses use seed 42. The corrected five-fold generator rerun uses seeds 42--46, with split and classifier seeds fixed at 42. Its environment includes Python 3.9.6, TensorFlow 2.15.0, NumPy 1.26.4, and scikit-learn 1.5.2. The run completed 15 generator fits at the original epoch settings. Fold membership, scaling extrema, IQR bounds, synthetic records, and individual fold AUCs are retained. Fixed seeds alone do not ensure identical results across software versions or hardware.',
        'This table contains the most counterintuitive': "The VAE has small absolute standardised mean differences on all 11 variables (maximum 0.230), but it is not the closest method for every variable. For example, its Albumin and SGOT differences exceed those of cGAN. Small mean differences should not be interpreted as agreement of the entire distribution.",
        'The explanation is that a variational': 'One possible interpretation is that concentration around central values preserves means while reducing variability. This is not a demonstrated mechanism or an inevitable property of VAEs. The KS and Jensen--Shannon results reveal distributional differences not captured by standardised mean differences.',
        'The practical lesson generalises': 'The results illustrate that quality rankings depend on the selected criterion. Mean agreement, distributional-shape agreement, and predictive usefulness are different properties and should be measured separately. This dataset does not establish a universal ranking of generative models.',
        'Fig.~\\ref{figS:tsne} projects': 'Figure~\\ref{figS:tsne} provides an exploratory two-dimensional projection of real and synthetic records. Apparent overlap or concentration in this projection does not establish high-dimensional fidelity, a causal explanation for privacy risk, or record-level disclosure. Those questions require separate quantitative analyses.',
        'The filtered VAE pool carries': 'The filtered VAE pool has a 35.7\\% near-duplicate rate under the specified distance threshold. This is an empirical proximity indicator, not a probability of re-identification. Table~\\ref{tab:perturb} reports post-hoc Gaussian perturbation of the eleven continuous variables, with noise measured in training-standard-deviation units.',
        'Fig.~\\ref{figS:prcurves} gives': 'Figure~\\ref{figS:prcurves} compares held-out precision--recall curves. Their interpretation depends on the event prevalence in the evaluation cohort, not only training-set balance. Visual proximity of curves does not demonstrate equivalence or establish a null effect.',
        'Fig.~\\ref{figS:auc} compares': 'Figure~\\ref{figS:auc} compares held-out AUC across Scenarios A--C. All displayed AUCs exceed 0.80, but this threshold does not establish clinical adequacy or prove that augmentation cannot harm a model. The changes differ by classifier and should be interpreted with their uncertainty.',
        'Of the 276 complete-case patients': 'Of 276 complete cases, 111 died, 18 received a transplant, and 147 were censored alive. Figure~\\ref{figS:competingrisks} displays Aalen--Johansen cumulative incidence with transplantation treated as a competing event. Estimated end-of-follow-up incidence is 64.2\\% for death and 8.7\\% for transplantation. This estimator accounts for competing events but still requires appropriate assumptions about other right censoring. The transplant frequency alone does not establish negligible effects on risk prediction. Cause-specific hazards and cumulative incidence answer different questions; a competing-risks regression was not fitted.',
        'Every quantity plotted in this section': 'These plots display the saved held-out classification results, which were reproduced at their reported precision in the artifact audit. They are not cross-validation plots; the corrected fold-contained results are reported separately in the main manuscript.',
    }
    for beginning, replacement in paragraphs.items():
        text = re.sub(r'(?m)^'+re.escape(beginning)+r'[^\n]*', lambda m: replacement, text)
    text = text.replace('Output dimension      & 18           & 18                & 18', 'Output dimension      & 19           & 19                & 19')
    text = text.replace('Section~IV', 'Section~4').replace('Section~III', 'Section~3')
    text = text.replace('Table~II of the main paper', 'the disclosure-risk table in the main paper')
    text = text.replace('near-duplicate rates in Table~II there', 'near-duplicate rates in its disclosure-risk table')
    text = text.replace('GAN (7.3\\%) and cGAN (9.6\\%) present low privacy risk, VAE the highest at 35.7\\%, and the equalised consensus set 21.1\\%.', 'The observed rates are 7.3\\% for GAN, 9.6\\% for cGAN, 35.7\\% for VAE, and 21.1\\% for consensus. Lower proximity rates do not certify safe release.')
    text = text.replace('which explains both its favourable distributional scores and its elevated near-duplicate rate of 21.1\\% reported in the disclosure-risk table in the main paper.', 'but the projection does not establish a causal explanation for the separately measured 21.1\\% near-duplicate rate.')
    text = text.replace('The VAE retains all generated records', 'The VAE retains 499 of 500 generated records')
    text = text.replace('{figS16_pipeline_flowchart.png}', '{figures/fig14_study_overview.png}')
    text = text.replace('End-to-end synthetic data pipeline. The raw 418-patient dataset passes through a complete-case filter and a preprocessing stage before the 193-patient training set feeds three parallel generative models. Each model\'s 500 generated records pass through an IQR plausibility filter before all three filtered pools enter the consensus voting stage.', 'Updated evaluation pipeline. The 276 complete cases are split into 193 training and 83 held-out patients before fitting preprocessing or generators. The corrected cross-validation independently refits generators, filters, and consensus inside each training fold; the test cohort is used only for evaluation.')
    text = text.replace('The masked-loss VAE and cGAN track the real curve closely; the VAE, CTGAN, and consensus diverge significantly, illustrating that FID does not isolate whether the joint time-to-event structure specifically is reproduced.', 'The comparisons are exploratory, unadjusted for multiplicity, and dependent on the reference cohort used for generation. Non-significant tests do not establish equivalent survival distributions.')
    text = text.replace('\\subsection{Reproducibility}', r'''\begin{figure}[tb]
\centering
\includegraphics[width=0.95\linewidth]{figS02_training_losses.png}
\caption{Saved full-training loss curves for the three core generative models. They describe optimisation, not demonstrated convergence, and are not loss curves from the corrected cross-validation rerun.}
\label{figS:training}
\end{figure}

\subsection{Reproducibility}''')
    text = text.replace(r'\resizebox{\columnwidth}{!}{', r'\fitwidth{')
    # Mark changed text and captions while preserving unchanged content.
    norm = lambda s: re.sub(r'\s+',' ',s).strip()
    original_norm = norm(original)
    lines = []
    for line in text.splitlines():
        if line.strip() and not line.lstrip().startswith(('%','\\')) and '&' not in line and norm(line) not in original_norm:
            line = r'\rev{'+line+'}'
        elif r'\caption{' in line and norm(line) not in original_norm:
            line = line.replace(r'\caption{',r'\caption{\revisioncolor ')
        elif 'Output dimension' in line:
            line = line.replace('Output dimension',r'\rev{Output dimension}').replace('19',r'\rev{19}')
        lines.append(line)
    text = '\n'.join(lines)
    # Accessible descriptions identify the supporting analysis without asserting
    # anything beyond the corresponding captions.
    text = re.sub(r'(\\includegraphics[^\n]*\{([^}]+)\})', lambda m:m.group(1)+'\n\\Description{Supporting figure: '+m.group(2).replace('.png','').replace('_',' ')+'. See the caption for the analysis and its limitations.}', text)
    supheader = header[:header.index(r'\begin{abstract}')]
    supheader = supheader.replace(r'\graphicspath{{figures/}}',r'\graphicspath{{figures_supplementary/}}')
    supheader = supheader.replace('Evaluating Synthetic Liver Cirrhosis Data:', 'Supplementary Material: Evaluating Synthetic Liver Cirrhosis Data:')
    supheader = supheader.replace(r'\begin{document}',r'\renewcommand{\thefigure}{S\arabic{figure}}'+'\n'+r'\renewcommand{\thetable}{S\arabic{table}}'+'\n'+r'\begin{document}')
    (PAPER/'supplementary.tex').write_text(supheader+'\n\\maketitle\n'+text+'\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{references}\n\\end{document}\n')
    (PAPER/'supplementary-clean.tex').write_text('\\def\\cleanversion{1}\n\\input{supplementary.tex}\n')


if __name__ == '__main__':
    prepare()
