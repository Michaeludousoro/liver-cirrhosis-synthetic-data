"""Targeted editorial revision of the supplied manuscript; no experiment execution.

Every substantive replacement is logged against the original source. Shared body
and conditional marking ensure that the blue and clean copies have identical text.
"""
from pathlib import Path
import re
import shutil
import json

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
OLD = HERE.parent / '2026-09-17/paper'
BASE = Path('/tmp/ieee-article-preview.xJzjKz/IEEE_Synthetic Data_latex/paper')
PAPER = HERE / 'paper'
PAPER.mkdir(parents=True, exist_ok=True)
original = (BASE / 'main.tex').read_text()
donor = (OLD / 'content.tex').read_text()
body = original[original.index(r'\section{Introduction}'):original.index(r'\begin{thebibliography}')]
ledger = []

def change(old, new, reason, category='Technical accuracy', count=None):
    global body
    n = body.count(old)
    if not n or (count is not None and n != count):
        raise ValueError(f'Replacement count {n}: {old[:100]}')
    marked = r'\rev{' + new + '}' if new else ''
    # Floats and full blocks carry their own revision marking.
    if new.startswith('BLOCK:'):
        marked = new[6:]
    body = body.replace(old, marked)
    ledger.append(dict(category=category, reason=reason, original=old, replacement=new.removeprefix('BLOCK:'), occurrences=n))

def paragraph(prefix, new, reason, category='Technical accuracy'):
    start = body.index(prefix)
    end = body.find('\n\n', start)
    if end == -1:
        end = len(body)
    change(body[start:end], new, reason, category, 1)

def donor_paragraph(prefix):
    start = donor.index(prefix)
    end = donor.index('\n\n', start)
    value = donor[start:end]
    if value.endswith('}'):
        value = value[:-1]
    return value

def donor_float(label):
    pos = donor.index(r'\label{' + label + '}')
    start = donor.rfind(r'\begin{', 0, pos)
    # The table label occurs before its tabular environment.
    env = re.match(r'\\begin\{([^}]+)\}', donor[start:]).group(1)
    end = donor.index(r'\end{' + env + '}', pos) + len(r'\end{' + env + '}')
    return donor[start:end]

# Reviewer markings are resolved individually, not silently stripped.
change(r'{\color{red} \st{Machine learning models built from these records have already shown good performance in predicting mortality among patients with liver cirrhosis}}',
       'Machine learning has also been investigated for cirrhosis mortality prediction', 'Reframe the struck-through claim without asserting universal performance.', 'Professor comments')
change(r'{\color{red}For these systems to work reliably}', 'Developing and evaluating these systems', 'Clarify the marked transition.', 'Professor comments')
change(r'{\color{red} \st{This study addresses that gap directly.}} ', '', 'Remove the struck-through sentence.', 'Professor comments')
change(r'{\color{red}random seed 42}', 'random seed 42', 'Retain the user-specified split seed.', 'Professor comments')
change(r' {\color{red} \st{that generates records based on patient outcome labels}}', '', 'Remove the struck-through duplication; conditioning remains in Methods.', 'Professor comments')
change('and VAE.\n\nIn the pre-processing', 'BLOCK:and VAE.\n\n'+r'\rev{A random seed sets the starting state of random-number generation, helping reproduce random choices. The value 42 is an arbitrary reproducibility setting, not a parameter chosen to improve accuracy. The split and classifiers use seed 42; the corrected cross-validation uses generator seeds 42--46 across its five folds.}'+'\n\nIn the pre-processing', 'Add the requested short seed-42 explanation and distinguish generator-fold seeds.', 'User request')
change(r'{\color{red}The results show that the proposed consensus voting method did not improve prediction accuracy, cross-validation stability, or disclosure risk for this small clinical dataset. Although this is a negative result, it is still valuable because it provides useful evidence for the design of future synthetic-data augmentation pipelines.[??IS SHOWING QUITE NEGATIVE IMPACT]}',
       'This evaluation distinguishes distributional quality from predictive utility and tests whether agreement across generators is an effective selection criterion. The findings inform task-specific use of synthetic clinical data without assuming that improved resemblance guarantees improved prediction.', 'Replace dismissive framing with the actual contribution, without changing the results.', 'Professor comments')
related = donor_paragraph('Two studies provide complementary methodological context.')
change(r'{\color{red}[?? CREATE A COMPARISON TABLE FOR BETTER IMPACT FOR ALL THE RELEVANT REFERENCES]}',
       'BLOCK:'+r'\rev{'+related+'}\n\n'+donor_float('tab:related'), 'Add the requested related-work comparison and relevant JDIQ/ACM HEALTH references.', 'Professor comments')
change(r'{\color{red}no. of follow-up days from registration to the event}', 'follow-up duration from registration to death, transplant, or censoring', 'Define follow-up duration precisely.', 'Professor comments')
change(r'{\color{red}alkaline phosphatase}', 'alkaline phosphatase', 'Retain and mark the requested expanded name.', 'Professor comments')
change(r'{\color{red} serum glutamic-oxaloacetic transaminase}', 'serum glutamic-oxaloacetic transaminase', 'Retain and mark the expanded name.', 'Professor comments')
change(r'{\color{red}[??VISUALISE THE DATA USING ADVANCE PLOTS LIKE RAINCLOUD]}',
       'BLOCK:'+r'\rev{Figure~\ref{fig:raincloud} shows selected biomarker distributions in the training cohort using densities, box plots, and individual observations.}'+'\n\n'+donor_float('fig:raincloud'), 'Insert the requested training-only raincloud plot.', 'Professor comments')
change(r'{\color{red}where, }', 'where $G$ is the generator, $D$ is the discriminator, $\\mathbf{x}$ is a real training record, and $\\mathbf{z}$ is a sampled noise vector. Expectations are over the real-data and latent-noise distributions, respectively.', 'Define the GAN equation symbols.', 'Professor comments', 2) if False else None
# Both equations had the same placeholder; replace each at its own position.
pos = body.index(r'{\color{red}where, }')
body = body[:pos]+body[pos:].replace(r'{\color{red}where, }', 'GAN_SYMBOLS_PENDING', 1)
change('GAN_SYMBOLS_PENDING', r'where $G$ is the generator, $D$ is the discriminator, $\mathbf{x}$ is a real training record, and $\mathbf{z}$ is a sampled noise vector. Expectations are over the real-data and latent-noise distributions, respectively.', 'Define the marked GAN equation symbols.', 'Professor comments')
change(r'{\color{red}and we name it accordingly:}', 'and is therefore labelled cGAN:', 'Use direct third-person architecture naming.', 'Professor comments')
change(r'As with the cGAN, {\color{red}we name it for what it is:[??WHAT YOU WANT TO EMPHASIZE]}', 'The distinction is architectural:', 'Clarify why the implementation is called VAE, not TVAE.', 'Professor comments')
change(r'{\color{red}and results reported here should not be read as a benchmark of that model.}', 'so the results do not benchmark TVAE.', 'Clarify the scope of the model comparison.', 'Professor comments')
change(r'{\color{red}Tukey fence filter [??REFERENCE]}', r'Tukey fence filter \cite{nistfences}', 'Supply the requested filtering reference.', 'Professor comments')
change(r'{\color{red}The present work differs in that three structurally distinct architectures are used, and acceptance requires cross-architecture agreement rather than simply pooling all outputs.[??EXPLAIN MORE]}',
       'The present work uses three structurally distinct architectures. For example, a GAN candidate must have a nearby record in both the cGAN and VAE pools; simple pooling would accept it without either check. Agreement concerns proximity in feature space, not reproduction of the same patient.', 'Explain cross-model agreement with a concrete example.', 'Professor comments')
change(r'A {\color{red}StandardScaler[??REFERENCE \& FIND BETTER WAY TO REPRESENT]} is fitted on the union of all three filtered datasets.',
       r'Each generated column is standardised as $z_j=(x_j-\mu_j)/s_j$, where $\mu_j$ and $s_j$ are the mean and standard deviation fitted to the union of the filtered training-derived pools using StandardScaler \cite{pedregosa2011}. Consensus distances use all 19 generated columns, including duration and status.', 'Give the standardisation equation, reference, and dimensions.', 'Professor comments')
change(r'{\color{red}where, }', r'where $\boldsymbol{\mu}_r$ and $\boldsymbol{\mu}_s$ are the real and synthetic mean vectors, $\boldsymbol{\Sigma}_r$ and $\boldsymbol{\Sigma}_s$ are their covariance matrices, and $\mathrm{Tr}$ denotes the matrix trace.', 'Define the marked FID equation symbols.', 'Professor comments')
change(r'{\color{red}Lower values indicate greater distributional alignment. Per-feature Kolmogorov-Smirnov (KS) tests, Jensen-Shannon divergence, Cohen\'s $d$, Shapiro-Wilk normality tests, and chi-square tests for categorical features provide a granular distributional profile.[?? ELABORATE APPROPRIATELY]}'.replace("\\'s", "'s"),
       r"Lower FID values indicate closer first- and second-moment agreement. Per-feature Kolmogorov-Smirnov tests compare cumulative distributions; Jensen-Shannon divergence describes distributional differences; Cohen's $d$ measures standardised mean differences. Shapiro-Wilk tests assess normality, while chi-square tests compare categorical frequencies. These complementary measures do not by themselves establish predictive utility.", 'Explain the requested quality metrics briefly.', 'Professor comments')
change(r'{\color{red}StandardScaler[??]}-normalised', 'standardised (using means and standard deviations fitted on real training patients)', 'Define the privacy-distance scaling reference.', 'Professor comments')
change(r'{\color{red}since it is needed only for the landmark and survival framings in Sections~\ref{sec:landmark} and~\ref{sec:survival}, but no classifier predicting status ever sees it.[??REWRITE]}',
       r'for outcome construction and survival modelling (Sections~\ref{sec:landmark} and~\ref{sec:survival}), but it is excluded from the predictors used to classify status.', 'Clarify generating duration versus using it as a predictor.', 'Professor comments')
change(r'{\color{red}classical baseline[?? WHY TWO BASELINES]}', 'classical oversampling comparator; Scenario A is the real-only reference', 'Distinguish the two comparator roles.', 'Professor comments')
paragraph('The classifiers use fixed configurations',
          'The classifiers use fixed configurations (RF and GB with 200 estimators, GB depth 3, LR with 1000 iterations) applied identically across all scenarios. Evaluation metrics are accuracy, F1, precision, recall, and AUC-ROC. The held-out evaluation is complemented by 1000-resample bootstrap confidence intervals and McNemar\'s exact test. '+donor_paragraph('Five-fold stratified validation partitions'),
          'Replace the incomplete validation description with the completed fold-contained protocol and software versions; resolve the marked Python-version query.', 'Validation correction')
change(r'{\color{red}Stating that precisely rather than just qualitatively would need a full competing-risks regression, for example a fine-gray model, which no package in our pipeline currently supports, so we did not fit one here.[??NOT CLEAR \& ALWAYS USE 3RD PERSON GRAMMAR]}',
       'A Fine--Gray regression was not fitted; the present analysis therefore does not estimate covariate effects on the cumulative incidence of death in the presence of transplantation.', 'Clarify the unperformed competing-risk analysis in third person.', 'Professor comments')
change(r'{\color{red}[?? CREATE A ABSTRACT DIAGRAM FOR BETTER IMPACT FOR ALL THE PROPOSED ALGORITHM]}',
       'BLOCK:'+donor_float('fig:overview'), 'Include the requested overview of the existing pipeline.', 'Professor comments')
change(r'{\color{red}Table~\ref{tab:performance} reports complete test-set metrics across all five scenarios.[?? EXPLAIN THIS TABLE A LITTLE BIT MORE BRIEFLY]}',
       r'Table~\ref{tab:performance} compares real-only training (A), augmentation (B--D), and synthetic-only training (E), all on the same 83 real test patients. AUC changes depend on the classifier: LR reaches 0.8467 with cGAN augmentation versus 0.8424 at baseline, while synthetic-only cGAN training gives AUC 0.8485 for LR but 0.6867 for GB.', 'Briefly explain how to read the performance table.', 'Professor comments')
paragraph(r'{\color{red}Extending scenario E', donor_paragraph('Under the three-year endpoint,'), 'Clarify why one-class VAE landmark results are unavailable.', 'Professor comments')
change(r'{\color{red}We did not run a significance test on the C-index, so read this single number the same way as the SMOTE result in the classification framing, a nominal gain we cannot yet confirm, not a contradiction of the null result that holds everywhere else.[??NOT REQUIRED]}',
       '', 'Remove the marked unnecessary passage; retain the short inference limitation in Threats to Validity.', 'Professor comments')
change(r'{\color{red}hypothesis[?? SOMEWHERE IN THE ABSTRACT ASSERT THIS AS HYPOTHESIS]}', 'relationship between fidelity and task-specific utility', 'State the augmentation hypothesis in the abstract; use precise local wording here.', 'Professor comments')
change(r'{\color{red}which output noise at $\sigma=0.7$ brings down to 1.4\% while also improving FID.[??NOT VERY CLEAR]}',
       r'which falls to 1.4\% after Gaussian noise with standard deviation $0.7$ times each continuous variable\'s training standard deviation is added to generated values. This is a distance-based result, not a privacy guarantee.'.replace("\\'s", "'s"), 'Explain the noise scale and the meaning of the near-duplicate reduction.', 'Professor comments')

# Local factual repairs: retain the surrounding original wording and structure.
local = [
('Hospitals and clinics that use internet of medical things (IoMT) devices to continuously collect measurements from wearable sensors allows them to monitor neurological disruptions under medically-defined specific health conditions, which are used by decision-support systems to help clinicians detect patient health deterioration', 'Clinical decision-support research has used electronic health records to identify cirrhosis and predict mortality', 'Do not attribute wearable neurological monitoring to EHR studies.'),
('An IoMT decision-support system is rarely trained and deployed at the same site: a model developed at one hospital is expected to run at another, on that institution\'s own patients and its own devices.', 'Cross-site deployment may require a model developed at one hospital to work on another institution\'s patients and devices.', 'Remove an unsupported universal deployment claim.'),
('Pooling records across sites would solve the sample-size problem, but it is precisely what data protection law prevents.', 'Pooling records may increase sample size, but requires appropriate data-governance arrangements.', 'Do not present all inter-hospital pooling as prohibited.'),
('and it is this cross-site question, rather than augmentation in the abstract, that motivates the present study.', 'although the present study evaluates augmentation and synthetic-only training within one cohort, not transfer between hospitals.', 'Align the motivation with the actual evaluation.'),
('but do not contain any actual patient information', 'but require separate assessment for memorisation and disclosure risk', 'Remove an unsupported privacy guarantee.'),
('as a secure method for sharing healthcare data without risking the exposure of personal patient information', 'as a possible method for sharing healthcare data while managing disclosure risk', 'Synthetic generation alone does not guarantee privacy.'),
('extended to six generators for the synthetic-only evaluation', 'extended to six synthetic-data methods (five generators and a derived consensus set) for the synthetic-only evaluation', 'Consensus is not a separately trained sixth generator.'),
('and shows across six generators that FID does not significantly predict train-on-synthetic-test-on-real utility.', 'and examines the association between FID and train-on-synthetic-test-on-real utility across six methods.', 'Report the association as an evaluation, not proof of no relationship.'),
('under three framing, leakage-free binary classification', 'under three framings: binary classification excluding follow-up duration', 'Avoid claiming removal of all possible leakage.'),
('so that the choice to binarise a censored outcome is not itself driving the conclusions.', 'to examine sensitivity to the outcome definition.', 'Different outcome analyses do not prove absence of censoring bias.'),
('This reminder of the paper', 'The remainder of the paper', 'Correct a local typographical error.'),
('classic augmentation methods have its own limitations', 'classic augmentation methods have their own limitations', 'Correct local agreement.'),
('In contrast, deep generative models handle these limitations better, since they learn those relationships directly from the data, and they are now one of the most common ways to generate synthetic clinical tabular data', 'Deep generative models can learn nonlinear relationships directly from data, although this flexibility does not guarantee better utility or fidelity', 'Remove unconditional model-superiority wording.'),
('While VAE uses a different strategy; instead of adversarial training,', 'A VAE uses a different strategy: instead of adversarial training,', 'Repair an incomplete sentence.'),
('However good distributional similarity', 'However, good distributional similarity', 'Correct punctuation.'),
('19 relevant features. The 19 features consist of 11 continuous laboratory measurements that includes,', '19 variables, including the outcome status. Of these, 11 are continuous variables, including', 'Distinguish generated columns from predictors and non-laboratory variables.'),
('sigmoid output of dimension 18', 'sigmoid output of dimension 19 (18 variables plus status)', 'Match both generator-output dimensions to the artifact.'),
('The adversarial losses settle without the oscillation that signals mode collapse, and the VAE ELBO falls sharply before levelling off by roughly epoch 50, so all three models are trained to convergence.', 'The loss curves describe optimisation under the stated epoch budgets; they do not establish convergence or rule out mode collapse.', 'Loss plots do not establish convergence.'),
('A single out-of-range value disqualifies the entire record.', r'A single out-of-range value disqualifies the entire record. Here $\mathrm{IQR}=Q_3-Q_1$; these statistical bounds do not certify clinical plausibility.', 'Define IQR and limit the meaning of the filter.'),
('the point at which class balance becomes acceptable and the quality improvement curve begins to flatten', 'an empirical trade-off between retained class balance and distributional similarity, not an independently validated optimum', 'Do not portray the selected tolerance as objectively optimal.'),
('tabular proxy FID computed on the 18-dimensional feature vectors', 'tabular proxy FID computed on the 11 continuous variables, including follow-up duration', 'Match the implemented FID columns.'),
('over the full 18-dimensional feature space', 'over the 11 continuous variables', 'Match the implemented FID columns.'),
('over the full 18-dimensional feature geometry', 'over the 11 continuous variables', 'Match the implemented FID columns.'),
('because a short N\\_days is tied to a recorded death events which is missing for all the patients.', 'because it includes post-enrolment follow-up information that is unavailable at prediction time.', 'Explain duration leakage without implying all short durations are deaths.'),
('without considering the observability of the event.', 'with event indicators distinguishing deaths from censored observations.', 'Survival fitting does use censoring indicators.'),
('which does not need a non-informative-censoring assumption.', 'which accounts for competing events while still requiring appropriate assumptions about other censoring.', 'Correct the Aalen--Johansen assumption claim.'),
('So the competing event is enough that censoring at transplant is unlikely to seriously bias our results.', 'These proportions alone do not establish that censoring at transplant has negligible impact on interpretation.', 'Frequency alone cannot exclude competing-risk effects.'),
('which confirms that smooth variational generation loses distributional sharpness.', 'indicating a larger mismatch in feature means and covariances.', 'FID alone does not identify a causal failure mechanism.'),
('fall at a safe distance from any real patient', 'lie above this empirical distance threshold', 'Distance is not a safety guarantee.'),
('reflecting the broader coverage of the variational latent space and the fact that consensus acceptance pools records lying close to real patients.', 'indicating greater proximity to training patients under this metric.', 'Avoid an untested causal explanation of proximity.'),
('it is far riskier than the filtered GAN (7.3\\%) or cGAN (9.6\\%), because it inherits the VAE records that reside closest to real patients.', 'it has a higher near-duplicate rate than filtered GAN (7.3\\%) or cGAN (9.6\\%); these rates are not probabilities of re-identification.', 'Distinguish a disclosure indicator from measured attack risk.'),
('but both reside well inside the overlapping bootstrap intervals in Table~\\ref{tab:significance} and neither McNemar comparison reaches significance value (Table~\\ref{tab:significance}), likely due noise effect rather than a real augmentation effect.', 'but neither paired accuracy comparison is significant (Table~\\ref{tab:significance}). McNemar tests accuracy, not AUC; overlapping individual AUC intervals are not a paired test of their difference.', 'Separate AUC estimation from paired accuracy inference.'),
('every scenario and classifier combination clears AUC~$= 0.8$, so no augmentation strategy damages discrimination, but none uniformly improves on the baseline either.', 'all classifier combinations in scenarios A--D exceed AUC~$=0.8$, but none of the augmentation strategies uniformly improves on baseline. Synthetic-only GB is lower at 0.6867.', 'Correct the false all-scenarios AUC claim.'),
('This fits a fixed-horizon task simply being easier to discriminate than an unbounded one; it is not evidence of a stronger augmentation effect.', 'The endpoint and eligible patient subset differ, so the higher AUCs do not by themselves indicate a stronger augmentation effect.', 'Do not infer task difficulty from non-equivalent outcomes.'),
('augmentation gives no statistically distinguishable improvement, while also avoiding the coarser censoring problem the reviewer raised.', 'no paired accuracy difference is significant. Excluding patients censored before three years addresses unknown labels but can introduce selection bias.', 'Qualify the censoring and statistical interpretation.'),
(', close to the classification framing\'s baseline AUC once N\\_days is excluded, which tells us that discarding censoring information was not itself hiding a lot of extra signal.', '; C-index and classification AUC are different measures and should not be interpreted as interchangeable.', 'Remove an invalid cross-metric inference.'),
('But none of that movement is distinguishable from noise (Table~\\ref{tab:significance}).', 'The paired accuracy comparisons are not significant (Table~\\ref{tab:significance}); this does not establish equivalence or test AUC differences.', 'Nonsignificance does not demonstrate noise or equivalence.'),
('So the null result holds across the main clinical groups in this cohort, not just overall.', 'These descriptive subgroup comparisons do not establish subgroup equivalence.', 'Avoid subgroup significance claims without a corresponding test.'),
('The masked-loss VAE has the best FID (0.062) and a strong mean TSTR AUC (0.798), which fits good utility.', 'The masked-loss VAE has the lowest FID (0.062) and mean TSTR AUC 0.798.', 'Remove an unsupported qualitative utility ranking.'),
('No steady relationship survives once a sixth generator joins the comparison. This six-point result is what our narrower abstract claim now rests on, not a guess from a single comparison.', 'With only six related methods, the association is inconclusive; these results do not establish the absence of a fidelity--utility relationship.', 'State the limited inference from six method-level observations.'),
('reproduce the real survival curve closely (log-rank', 'show no significant log-rank difference from the real curve (', 'A nonsignificant log-rank result does not prove curve equivalence.'),
('N\\_days, bilirubin, and albumin, which Fig.~\\ref{fig:importance} ranks as the first, second, and fourth most important predictors.', 'N\\_days, bilirubin, and albumin. N\\_days is an outcome duration rather than a baseline predictor.', 'Remove the obsolete duration-inclusive importance ranking.'),
('The RF gain is the closest thing to a real effect anywhere in this study,', 'The RF comparison has', 'Retain the favourable exploratory result without declaring a real effect.'),
('but McNemar\'s test still falls short of significance ($p = 0.125$) and the bootstrap intervals still overlap.', 'but the paired accuracy comparison is not significant (McNemar $p=0.125$); this is not a test of the AUC gain.', 'Separate the positive AUC estimate from its accuracy p-value.'),
('the proposed test is simply underpowered to confirm an effect this size at $n=83$.', 'small paired differences are difficult to resolve at $n=83$; the power calculation concerns accuracy, not AUC.', 'Do not apply an accuracy-power calculation to an AUC effect.'),
('noise-level changes', 'small descriptive changes', 'Avoid declaring an untested change to be noise.'),
('told a different theory', 'showed a different pattern', 'Describe observations rather than claiming a new theory.'),
('a tabular proxy on 18-dimensional feature vectors', 'a tabular proxy on the 11 continuous variables', 'Correct the FID dimension consistently.'),
('and the proposed DP-SGD analysis shows a formal privacy guarantee is out of reach at this sample size anyway.', 'and the simplified DP-SGD projections do not establish a formal privacy guarantee.', 'Do not infer impossibility of differential privacy from a projection.'),
('Fig.~S11 in the supplementary file suggests this is unlikely to bias the results much, since transplant makes up only 6.5\\% of the cohort against 40.2\\% for death, but confirming that properly would need a Fine-Gray model, which no package in our pipeline currently supports.', 'Fig.~S11 describes the competing-event incidence, but does not establish negligible bias. A Fine--Gray regression and paired C-index significance tests were not performed.', 'Retain the survival limitations accurately and concisely.'),
('so it should not be read as a benchmark of cGAN as originally published.', 'so it should not be read as a benchmark of CTGAN as originally published.', 'Correct the comparator model name.'),
('the cGAN is the safest choice here: lowest FID paired with a low near-duplicate rate.', 'the cGAN combines the lowest FID among the three core models with a relatively low near-duplicate rate, but release safety requires a separate assessment.', 'Do not turn relative metrics into a deployment recommendation.'),
('and this same pattern showed up for all six generators, not just cGAN.', 'illustrating why synthetic-only utility must be checked separately for each generator and classifier.', 'Remove a pattern not shared by every method.'),
('Every quantity plotted in S6 to S10 also appears numerically in Tables~\\ref{tab:performance} and~\\ref{tab:cv}.', 'These historical supplementary plots should not be used as evidence for the corrected fold-contained cross-validation; its results are reported in Table~\\ref{tab:cv}.', 'Distinguish historical plots from the rerun validation.'),
("Cohen's $d$ ranks the VAE as the best-matched method on every feature even though it is the worst on every distributional-shape measure, which is a further instance of the paper's central point that a single fidelity metric can be satisfied by a model that is wrong in every other respect.", "the VAE has small absolute standardised mean differences despite substantial distributional-shape differences. It is not the closest method on every individual mean-difference measure, reinforcing the need for complementary quality criteria.", 'Correct the per-feature ranking overstatement.'),
]
for a,b,why in local:
    change(a,b,why)

# Replace only paragraphs whose central technical claim must change.
paragraph('Table~\\ref{tab:cv} presents', donor_paragraph('Table~\\ref{tab:cv} replaces'), 'Insert the verified fold-contained validation interpretation.', 'Validation correction')
for a,b in [('0.8476 \\pm 0.0537','0.8227 \\pm 0.0573'),('0.8169 \\pm 0.0586','0.8081 \\pm 0.0688'),('0.8357 \\pm 0.0502','0.8184 \\pm 0.0618'),('0.8295 \\pm 0.0532','0.8377 \\pm 0.0723'),('0.7880 \\pm 0.0480','0.8088 \\pm 0.0573'),('0.8349 \\pm 0.0523','0.7803 \\pm 0.0404')]:
    change('$'+a+'$', '$'+b+'$', 'Replace an invalid generative CV estimate with its verified rerun mean and fold SD.', 'Validation correction')
change('Mean $\\pm$ Std, real-only validation folds, N\\_Days excluded', 'Mean $\\pm$ Fold SD, fold-contained generation, real-only validation, N\\_Days excluded', 'Specify corrected validation and avoid interpreting SD as a confidence interval.', 'Validation correction')
paragraph('The cross-validation results back this up', donor_paragraph('The corrected cross-validation results do not support'), 'Update Discussion to agree with the corrected CV values.', 'Validation correction')
paragraph('The full VAE privacy-enhancement sweep', donor_paragraph('The output-perturbation sweep is shown'), 'Correct the false claim that DP-SGD models were trained; keep the existing perturbation results.')
paragraph('A generative model trained on the 193', 'A generative model trained on the 193 real training patients is a function of those patients plus randomness. Its synthetic records do not add independent patient observations. Augmentation can nevertheless reweight or smooth the training distribution, regularise a classifier, or change class balance. The theory motivating this study is therefore that these changes may improve prediction; the empirical question is whether the proposed filtering and consensus procedure produces such a benefit on held-out real patients.', 'Keep the theory subsection and augmentation rationale, but remove the unsupported impossibility theorem.')
paragraph('The consensus mechanism actually works against itself here.', 'Cross-architecture agreement, by design, favours records represented in all three generated pools. The consensus set has a 21.1\\% near-duplicate rate, higher than either adversarial pool, suggesting that proximity and diversity warrant attention. Concentration in already represented regions is one possible explanation for the limited augmentation benefit, but these experiments do not establish that every accepted record is redundant or that only decision-boundary samples can improve learning.', 'Retain the proposed mechanism as a hypothesis rather than an established cause.')
change('This also explains why the effect differs by classifier.', 'Classifier sensitivity is another possible explanation for the differing effects.') if False else None
change('This also explains why the effect differs by classifier.', 'Classifier sensitivity is another possible explanation for the differing effects.', 'Identify the classifier mechanism as an interpretation.')
change('so points that fall inside an existing partition leave the split thresholds almost untouched.', 'and additional training points may alter split thresholds or leaf estimates.', 'Tree models can change even when records fall within existing partitions.')
change('which is why it shows the largest movement across scenarios in Table~\\ref{tab:performance}.', 'although the present experiments do not isolate this mechanism.', 'Remove the false largest-movement claim and unsupported causality.')
change('Synthetic data can substitute for real data that is missing, but it cannot add to real data that is already enough. That distinction, substitution rather than augmentation, is the practical lesson of this study.', 'Synthetic-only training therefore retains predictive signal in this split, while augmentation shows no consistent advantage. This does not establish equivalence to real data or a sample-size threshold beyond which augmentation cannot help.', 'Avoid generalising one cohort into a universal augmentation limit.')
paragraph('Scenario~E matters most for IoMT settings', 'Scenario~E is relevant to possible IoMT settings where access to real patient records is restricted. The results show that classifiers trained on synthetic cGAN records can retain predictive signal on held-out patients from the same source cohort. They do not demonstrate transfer between hospitals, reduced communication costs, or a compliant release mechanism. The cGAN near-duplicate rate of 9.6\\% and the perturbed VAE rate of 1.4\\% are empirical distance indicators, not negligible-risk or formal privacy guarantees. Cross-site use remains a motivation for future evaluation rather than a deployment validated in this study.', 'Keep the IoMT motivation but remove claims of experiments and guarantees not performed.')
paragraph('Two limits are worth stating here.', 'Two limits are worth stating here. First, synthetic-only cGAN training gave AUC 0.687 for GB, compared with 0.809 for RF and 0.849 for LR, so usefulness is classifier dependent. Second, the results come from one retrospective cohort; a receiving institution would need its own utility, privacy, and governance assessment before deployment.', 'Replace unsupported cross-generator and cross-site generalisations.')
paragraph('The study conducted in this paper suggests four crucial aspects', 'The study suggests four practical considerations for small tabular cohorts. First, validation must contain real patients only, and every fitted preprocessing, generation, filtering, and selection step must use only the corresponding training fold. Restricting validation to real records is insufficient if the generator has already seen those patients.', 'Make practical guidance match the corrected leakage mechanism.', 'Validation correction')
paragraph('The Mayo Clinic PBC dataset is publicly available from', donor_paragraph('The Mayo Clinic PBC dataset is publicly available from'), 'Clarify reproducibility scope and seed differences without claiming every analysis was rerun.')
paragraph('The authors used a generative AI assistant', donor_paragraph('The authors used a generative AI assistant'), 'Describe assistance without claiming unverified coauthor approval.')

# A few remaining local inference corrections, without rewriting the surrounding sections.
change('confirms synthetic', 'confirms synthetic', 'placeholder') if False else None
change('all confidence intervals overlap substantially across scenarios, including SMOTE, and all nine McNemar p-values exceed 0.06, confirming that', 'all nine McNemar p-values exceed 0.06, so', 'Overlapping AUC intervals are not the basis of McNemar accuracy inference.')
change('both are short of $\\alpha = 0.05$ and, per Section~\\ref{sec:power}, the test is underpowered at this sample size regardless.', 'both exceed $\\alpha=0.05$ and both comparisons have more newly incorrect than newly correct predictions. The small test set limits inference.', 'Do not present the two closest p-values as promising gains.')
change('The simpler model still outperforms;', 'Under this training budget, the simpler model has lower FID;', 'Do not infer overall superiority from fidelity alone.')
change('This gap looks like a property of the small cohort rather than a flaw in CTGAN itself:', 'The training budget may contribute to this gap:', 'Identify the convergence explanation as tentative.')
change('This is likely short of full convergence at this cohort size', 'Convergence was not established at this training budget', 'Do not assert unmeasured convergence status.')
change('a partially-converged snapshot', 'a fixed-budget snapshot', 'Use a verifiable training-budget description.')
change('so this exclusion follows trial protocol, not anything about the patients themselves.', 'so much of this missingness reflects protocol differences; complete-case selection can still introduce bias.', 'Protocol-related missingness does not establish absence of selection bias.')
change('across the six generators', 'across the six synthetic-data methods', 'Name the consensus-derived comparison correctly.')
change('all six generators', 'all six synthetic-data methods', 'Name the consensus-derived comparison correctly.')
change('Across Six Generators', 'Across Six Synthetic-Data Methods', 'Name the consensus-derived comparison correctly.')
change('as a sixth generator', 'as an additional generator alongside the consensus-derived set', 'Consensus is a derived set, not an independent generator.')
change('same seed-42 setup as every other generator', 'seed-42 setup used for the original full-training-pool generators', 'Distinguish historical generation seeds from corrected fold seeds.')

conclusion = r'''This paper evaluated what synthetic patient data can contribute to predicting liver cirrhosis outcomes using the Mayo Clinic PBC dataset. Of 418 records, 276 complete cases provided 193 training patients and 83 held-out test patients. Three core generative models---Vanilla GAN, cGAN, and VAE---were combined with IQR filtering and consensus voting. Additional comparisons included SMOTE, a masked-loss VAE using partially observed records, and CTGAN. Follow-up duration was excluded from the 17 baseline predictors, while landmark classification and survival modelling examined alternative outcome definitions.

The results distinguish distributional fidelity from predictive utility. The masked-loss VAE achieved the lowest tabular FID, 0.062, followed by cGAN at 0.072. However, cGAN and consensus augmentation did not consistently improve held-out discrimination. Corrected five-fold validation, with generators and filtering fitted independently within each training fold, produced lower mean AUCs for both strategies than real-only training across all three classifiers. SMOTE gave a small descriptive Random Forest increase, from 0.8499 to 0.8559. These fold summaries are not significance tests.

The masked-loss VAE produced a favourable exploratory Random Forest AUC change, from 0.832 to 0.868, while the paired accuracy comparison remained non-significant (McNemar $p=0.125$). Synthetic-only cGAN training retained predictive signal, reaching AUC 0.809 for Random Forest and 0.849 for Logistic Regression, compared with 0.687 for Gradient Boosting. These findings support classifier-specific evaluation, not equivalence to real-data training. The six-method fidelity--utility association was inconclusive, and survival C-index differences were descriptive rather than formally tested.

Consensus filtering retained 787 records but did not reduce the near-duplicate indicator relative to the adversarial pools: its rate was 21.1%, compared with 7.3% for GAN and 9.6% for cGAN. Output perturbation reduced the VAE indicator, but neither these distances nor the simplified DP-SGD projections establish a privacy guarantee. Statistical filtering should therefore complement, rather than replace, task-specific utility and disclosure assessment.

The principal limitations are the small single-cohort test set, complete-case selection, retrospective internal validation, and limited random-seed evaluation. Non-significant accuracy comparisons do not rule out modest benefits or harms. Recovering partially observed records remains a promising direction for further investigation. Larger external cohorts, repeated-seed validation, and direct cross-site evaluation are needed before drawing deployment conclusions for IoMT applications.'''
conclusion = conclusion.replace('%', r'\%')
change('The RF gain is the closest thing', 'The RF gain is the closest thing', '') if False else None
change('No scenario in this section overturns the paper\'s main null hypothesis result at conventional significance. But the masked-loss VAE result is the one place where the size and direction of the effect, not just its sign, plausibly reflects something real: a modest, classifier-specific benefit from using more of the real signal we already have, through masked training rather than throwing partial records away.', 'The masked-loss VAE results motivate further investigation of partially observed records. The favourable RF AUC estimate is exploratory, and the non-significant paired accuracy test neither confirms a benefit nor rules one out.', 'Preserve the promising partial-data finding without claiming that it reflects a proven effect.')
change('This does not undo the null hypothesis.', 'This conditional calculation does not establish whether a true effect exists.', 'Qualify the interpretation of the historical power calculation.')
change('a mean of 8 pairs, a 9.2\\% discordant rate on $n=83$', 'approximately eight discordant pairs, or roughly 9\\% of $n=83$', 'Avoid implying exact agreement of rounded discordance summaries.')
change('The main question about following the path from distributional fidelity to predictive utility has received much less attention, especially for small tabular clinical datasets used in IoMT environments. \\cite{kababji2023}, \\cite{espinosa2023}, and that is the main motivation of this study to test predictive utility on unseen real patients, not just similarity metrics.', 'These findings motivate testing predictive utility on unseen real patients alongside distributional similarity, particularly for small clinical datasets \\cite{kababji2023,espinosa2023}.', 'Repair the incomplete related-work transition without introducing a new concept.')
change('The trained VAE whose reconstruction loss only counts', 'A VAE was trained with a reconstruction loss that only counts', 'Repair the incomplete masked-loss methods sentence.')
change('The death has been encoded as event (1) and both staying alive and getting a transplant as non-events (0).', 'Death is encoded as event (1), while censoring and transplantation are encoded as non-events (0).', 'Distinguish censored status from known survival.')
start = body.index(r'{\color{red}[??CONCLUSION NEED TO BE BRIEF AND PRECISE]')
end = body.index(r'\subsection*{Supplementary Material}', start)
change(body[start:end], 'BLOCK:'+'\n\n'.join(r'\rev{'+p+'}' for p in conclusion.split('\n\n'))+'\n\n', 'Replace the repetitive conclusion with a 350--400-word summary of the existing study, findings and limitations.', 'Professor comments', 1)

# Unchanged diagrams/tables remain black; ACM width fitting is layout-only.
change(r'\rev{Developing and evaluating these systems}, they require large and well-labelled datasets, and obtaining such datasets is difficult.', r'Developing and evaluating these systems requires large and well-labelled datasets, which can be difficult to obtain.', 'Repair the grammar of the marked introductory transition.', 'Professor comments')
change('from 0.070 to 1.00', 'from 0.063 to 1.00', 'Make the stated p-value range match the unchanged significance table.')
change('and still gave no predictive benefit.', 'and still gave no consistent augmentation benefit.', 'Distinguish lack of a consistent benefit from absence of any predictive signal.')
change('training: 115 survived/transplanted', 'training: 115 censored/transplanted', 'Use the recorded outcome label rather than imply known long-term survival.')
for filename in ('fig06_roc_curves.png','fig07_feature_importance.png','fig08_power_analysis.png','fig11_fid_vs_tstr.png'):
    desc = re.search(r'\\includegraphics[^\n]*\{'+re.escape(filename)+r'\}\n(\\Description\{[^\n]+\})',donor)
    if desc:
        body = re.sub(r'(\\includegraphics[^\n]*\{'+re.escape(filename)+r'\})',lambda m:m[1]+'\n'+desc[1],body)
body = body.replace(r'\resizebox{\columnwidth}{!}{', r'\fitwidth{')
body = body.replace(r'\FloatBarrier', '')
assert r'\color{red}' not in body and '[??' not in body
body += '\n\\bibliographystyle{ACM-Reference-Format}\n\\bibliography{references}\n'
(PAPER/'content.tex').write_text(body)

header = (OLD/'main.tex').read_text()
title = re.search(r'\\title\{([^}]+)\}', original).group(1)
header = re.sub(r'\\title\[.*?\]\{.*?\}\n', lambda _: '\\title[Theory of Augmentation for Improved Prediction Accuracy]{'+title+'}\n', header, count=1)
header = re.sub(r'\\keywords\{[^}]*\}', lambda _: r'\keywords{\rev{Synthetic data generation; liver cirrhosis; predictive utility; consensus voting; patient privacy}}', header)
header = header.replace('Blue text identifies revised or added paragraphs, tables, captions, and reformatted references relative to the supplied IEEE manuscript. Unchanged prose remains black. Figures retain their original colours. Historical fold-leaked validation estimates have been replaced by a completed fold-contained rerun.', 'Targeted revision relative to the supplied manuscript. Blue text identifies corrected or added wording and updated values; unchanged wording remains black. Figures retain their original colours. The supplied title and study scope are retained. The accompanying change record documents removals and technical corrections.')
header = header.replace('Reviewer copy, 17 September 2026.', 'Targeted reviewer copy, 17 September 2026.')
(PAPER/'main.tex').write_text(header)
(PAPER/'clean.tex').write_text('\\def\\cleanversion{1}\n\\input{main.tex}\n')
for name in ('acmart.cls','ACM-Reference-Format.bst','references.bib','supplementary.tex','supplementary-clean.tex'):
    shutil.copy2(OLD/name, PAPER/name)
supp = (PAPER/'supplementary.tex').read_text()
supp = re.sub(r'\\title\[.*?\]\{.*?\}\n', lambda _: '\\title[Supplementary Material]{Supplementary Material: '+title+'}\n', supp, count=1)
(PAPER/'supplementary.tex').write_text(supp)
for folder in ('figures','figures_supplementary'):
    shutil.copytree(OLD/folder, PAPER/folder, dirs_exist_ok=True)
# Mark only the four added bibliography titles, not the unchanged bibliography.
bib = (PAPER/'references.bib').read_text()
for key in ('stenger2024','mamun2026','nistfences','pedregosa2011'):
    start = bib.index('{'+key+',')
    end = bib.index('\n}\n',start)
    entry = bib[start:end]
    entry = re.sub(r'(?m)^(  title = )\{\{(.*)\}\}(,?)$', lambda m:m[1]+'{{\\rev{'+m[2]+'}}}'+m[3],entry)
    bib = bib[:start]+entry+bib[end:]
(PAPER/'references.bib').write_text(bib)

def words(s):
    s = re.sub(r'\\[a-zA-Z]+\*?(?:\[[^]]*\])?', '', s)
    return len(re.findall(r"[A-Za-z0-9]+(?:[\-'][A-Za-z0-9]+)*",s))
abstract = re.search(r'\\begin\{abstract\}(.*?)\\end\{abstract\}',header,re.S).group(1)
counts = {'abstract':words(abstract),'conclusion':words(conclusion)}
assert 200 <= counts['abstract'] <= 250, counts
assert 350 <= counts['conclusion'] <= 400, counts
(HERE/'change-ledger.json').write_text(json.dumps(ledger,indent=2,ensure_ascii=False))
(HERE/'word-counts.json').write_text(json.dumps(counts,indent=2))
print('Built targeted manuscript:', len(ledger), 'logged replacements;',counts)
