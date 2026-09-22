# Detailed replacement ledger

Each entry records the original text, replacement, and reason. Layout-only ACM conversion is excluded. Abstract shortening, the retained title and five keywords are documented in the main correction record.

## 1. Professor comments

Reframe the struck-through claim without asserting universal performance.

Original:

```tex
{\color{red} \st{Machine learning models built from these records have already shown good performance in predicting mortality among patients with liver cirrhosis}}
```

Replacement:

```tex
Machine learning has also been investigated for cirrhosis mortality prediction
```

## 2. Professor comments

Clarify the marked transition.

Original:

```tex
{\color{red}For these systems to work reliably}
```

Replacement:

```tex
Developing and evaluating these systems
```

## 3. Professor comments

Remove the struck-through sentence.

Original:

```tex
{\color{red} \st{This study addresses that gap directly.}} 
```

Replacement:

```tex
[Removed]
```

## 4. Professor comments

Retain the user-specified split seed.

Original:

```tex
{\color{red}random seed 42}
```

Replacement:

```tex
random seed 42
```

## 5. Professor comments

Remove the struck-through duplication; conditioning remains in Methods.

Original:

```tex
 {\color{red} \st{that generates records based on patient outcome labels}}
```

Replacement:

```tex
[Removed]
```

## 6. User request

Add the requested short seed-42 explanation and distinguish generator-fold seeds.

Original:

```tex
and VAE.

In the pre-processing
```

Replacement:

```tex
and VAE.

\rev{A random seed sets the starting state of random-number generation, helping reproduce random choices. The value 42 is an arbitrary reproducibility setting, not a parameter chosen to improve accuracy. The split and classifiers use seed 42; the corrected cross-validation uses generator seeds 42--46 across its five folds.}

In the pre-processing
```

## 7. Professor comments

Replace dismissive framing with the actual contribution, without changing the results.

Original:

```tex
{\color{red}The results show that the proposed consensus voting method did not improve prediction accuracy, cross-validation stability, or disclosure risk for this small clinical dataset. Although this is a negative result, it is still valuable because it provides useful evidence for the design of future synthetic-data augmentation pipelines.[??IS SHOWING QUITE NEGATIVE IMPACT]}
```

Replacement:

```tex
This evaluation distinguishes distributional quality from predictive utility and tests whether agreement across generators is an effective selection criterion. The findings inform task-specific use of synthetic clinical data without assuming that improved resemblance guarantees improved prediction.
```

## 8. Professor comments

Add the requested related-work comparison and relevant JDIQ/ACM HEALTH references.

Original:

```tex
{\color{red}[?? CREATE A COMPARISON TABLE FOR BETTER IMPACT FOR ALL THE RELEVANT REFERENCES]}
```

Replacement:

```tex
\rev{Two studies provide complementary methodological context. Stenger et al.\ organise synthesis-quality assessment for time series \cite{stenger2024}; the present work applies a multidimensional evaluation perspective to a static clinical table with survival outcomes. Mamun et al.\ combine CTGAN augmentation with filtering for neonatal outcome prediction and explanation \cite{mamun2026}; the present study examines a different disease cohort and explicitly contrasts fidelity, augmentation, synthetic-only utility, and empirical disclosure indicators.}

\begin{table*}[!t]
\revisioncolor
\centering
\caption{\revisioncolor Selected Related Work and Its Relationship to the Present Evaluation}
\label{tab:related}
\footnotesize
\begin{tabular}{p{0.20\textwidth}p{0.28\textwidth}p{0.44\textwidth}}
\toprule
Study & Focus & Relationship to the present study \\
\midrule
Gon\c{c}alves et al.\ \cite{goncalves2020}; Gonzales et al.\ \cite{gonzales2023} & Synthetic patient data generation and healthcare synthesis literature & Establish the clinical-data context; the present study evaluates a specific PBC cohort and prediction task. \\
Choi et al.\ \cite{choi2017}; Xu et al.\ \cite{xu2019} & Generative modelling of patient records and mixed tabular data & Provide model context; cGAN, VAE, and CTGAN are distinguished explicitly rather than treated as interchangeable architectures. \\
Yan et al.\ \cite{yan2022} & Benchmarking synthetic electronic health records & Motivates reporting multiple evaluation dimensions alongside downstream prediction. \\
Stenger et al.\ \cite{stenger2024} (JDIQ) & Systematic assessment of time-series synthesis quality & Supports structured quality evaluation; the data modality differs from the static PBC table. \\
Mamun et al.\ \cite{mamun2026} (ACM HEALTH) & CTGAN augmentation and filtering for neonatal prediction and explanation & Provides a clinical augmentation comparison; benefits must be tested anew for the PBC task. \\
Present study & PBC synthesis, statistical filtering, and cross-model consensus & Separates fidelity, empirical disclosure risk, augmentation, and synthetic-only utility, with three outcome framings. \\
\bottomrule
\end{tabular}
\end{table*}
```

## 9. Professor comments

Define follow-up duration precisely.

Original:

```tex
{\color{red}no. of follow-up days from registration to the event}
```

Replacement:

```tex
follow-up duration from registration to death, transplant, or censoring
```

## 10. Professor comments

Retain and mark the requested expanded name.

Original:

```tex
{\color{red}alkaline phosphatase}
```

Replacement:

```tex
alkaline phosphatase
```

## 11. Professor comments

Retain and mark the expanded name.

Original:

```tex
{\color{red} serum glutamic-oxaloacetic transaminase}
```

Replacement:

```tex
serum glutamic-oxaloacetic transaminase
```

## 12. Professor comments

Insert the requested training-only raincloud plot.

Original:

```tex
{\color{red}[??VISUALISE THE DATA USING ADVANCE PLOTS LIKE RAINCLOUD]}
```

Replacement:

```tex
\rev{Figure~\ref{fig:raincloud} shows selected biomarker distributions in the training cohort using densities, box plots, and individual observations.}

\begin{figure*}[!t]
\revisioncolor
\centering
\includegraphics[width=0.95\textwidth]{fig13_training_raincloud.png}
\Description{Training-only biomarker distributions stratified by recorded status; points, densities, and box plots show the small-sample spread.}
\caption{\revisioncolor Raincloud plots of bilirubin, albumin, and prothrombin in the 193 real training patients, grouped by recorded outcome (78 deaths; 115 censored/transplanted). Each panel combines a half-violin density, median and interquartile range, and individual observations. Bilirubin uses a logarithmic axis; its density is estimated in log space. Censored/transplanted status does not imply survival over a common follow-up horizon.}
\label{fig:raincloud}
\end{figure*}
```

## 13. Professor comments

Define the marked GAN equation symbols.

Original:

```tex
GAN_SYMBOLS_PENDING
```

Replacement:

```tex
where $G$ is the generator, $D$ is the discriminator, $\mathbf{x}$ is a real training record, and $\mathbf{z}$ is a sampled noise vector. Expectations are over the real-data and latent-noise distributions, respectively.
```

## 14. Professor comments

Use direct third-person architecture naming.

Original:

```tex
{\color{red}and we name it accordingly:}
```

Replacement:

```tex
and is therefore labelled cGAN:
```

## 15. Professor comments

Clarify why the implementation is called VAE, not TVAE.

Original:

```tex
As with the cGAN, {\color{red}we name it for what it is:[??WHAT YOU WANT TO EMPHASIZE]}
```

Replacement:

```tex
The distinction is architectural:
```

## 16. Professor comments

Clarify the scope of the model comparison.

Original:

```tex
{\color{red}and results reported here should not be read as a benchmark of that model.}
```

Replacement:

```tex
so the results do not benchmark TVAE.
```

## 17. Professor comments

Supply the requested filtering reference.

Original:

```tex
{\color{red}Tukey fence filter [??REFERENCE]}
```

Replacement:

```tex
Tukey fence filter \cite{nistfences}
```

## 18. Professor comments

Explain cross-model agreement with a concrete example.

Original:

```tex
{\color{red}The present work differs in that three structurally distinct architectures are used, and acceptance requires cross-architecture agreement rather than simply pooling all outputs.[??EXPLAIN MORE]}
```

Replacement:

```tex
The present work uses three structurally distinct architectures. For example, a GAN candidate must have a nearby record in both the cGAN and VAE pools; simple pooling would accept it without either check. Agreement concerns proximity in feature space, not reproduction of the same patient.
```

## 19. Professor comments

Give the standardisation equation, reference, and dimensions.

Original:

```tex
A {\color{red}StandardScaler[??REFERENCE \& FIND BETTER WAY TO REPRESENT]} is fitted on the union of all three filtered datasets.
```

Replacement:

```tex
Each generated column is standardised as $z_j=(x_j-\mu_j)/s_j$, where $\mu_j$ and $s_j$ are the mean and standard deviation fitted to the union of the filtered training-derived pools using StandardScaler \cite{pedregosa2011}. Consensus distances use all 19 generated columns, including duration and status.
```

## 20. Professor comments

Define the marked FID equation symbols.

Original:

```tex
{\color{red}where, }
```

Replacement:

```tex
where $\boldsymbol{\mu}_r$ and $\boldsymbol{\mu}_s$ are the real and synthetic mean vectors, $\boldsymbol{\Sigma}_r$ and $\boldsymbol{\Sigma}_s$ are their covariance matrices, and $\mathrm{Tr}$ denotes the matrix trace.
```

## 21. Professor comments

Explain the requested quality metrics briefly.

Original:

```tex
{\color{red}Lower values indicate greater distributional alignment. Per-feature Kolmogorov-Smirnov (KS) tests, Jensen-Shannon divergence, Cohen's $d$, Shapiro-Wilk normality tests, and chi-square tests for categorical features provide a granular distributional profile.[?? ELABORATE APPROPRIATELY]}
```

Replacement:

```tex
Lower FID values indicate closer first- and second-moment agreement. Per-feature Kolmogorov-Smirnov tests compare cumulative distributions; Jensen-Shannon divergence describes distributional differences; Cohen's $d$ measures standardised mean differences. Shapiro-Wilk tests assess normality, while chi-square tests compare categorical frequencies. These complementary measures do not by themselves establish predictive utility.
```

## 22. Professor comments

Define the privacy-distance scaling reference.

Original:

```tex
{\color{red}StandardScaler[??]}-normalised
```

Replacement:

```tex
standardised (using means and standard deviations fitted on real training patients)
```

## 23. Professor comments

Clarify generating duration versus using it as a predictor.

Original:

```tex
{\color{red}since it is needed only for the landmark and survival framings in Sections~\ref{sec:landmark} and~\ref{sec:survival}, but no classifier predicting status ever sees it.[??REWRITE]}
```

Replacement:

```tex
for outcome construction and survival modelling (Sections~\ref{sec:landmark} and~\ref{sec:survival}), but it is excluded from the predictors used to classify status.
```

## 24. Professor comments

Distinguish the two comparator roles.

Original:

```tex
{\color{red}classical baseline[?? WHY TWO BASELINES]}
```

Replacement:

```tex
classical oversampling comparator; Scenario A is the real-only reference
```

## 25. Validation correction

Replace the incomplete validation description with the completed fold-contained protocol and software versions; resolve the marked Python-version query.

Original:

```tex
The classifiers use fixed configurations (RF and GB with 200 estimators, GB depth 3, LR with 1000 iterations) applied identically across all scenarios so the comparison is fair. Evaluation metrics are accuracy, F1, precision, recall, and AUC-ROC. Five-fold stratified cross-validation, 1000-resample bootstrap confidence intervals, and McNemar's exact test complement the single-split evaluation. In cross-validation the folds are drawn from the real training patients only; synthetic records are added exclusively to the training side of each fold and every validation fold contains real patients alone, making validation process robust against synthetic record leakage. All random seeds are fixed at 42, and the generative models are trained under fixed Python{\color{red}[??VERSION]}, NumPy, and TensorFlow seeds so the full pipeline is reproducible.
```

Replacement:

```tex
The classifiers use fixed configurations (RF and GB with 200 estimators, GB depth 3, LR with 1000 iterations) applied identically across all scenarios. Evaluation metrics are accuracy, F1, precision, recall, and AUC-ROC. The held-out evaluation is complemented by 1000-resample bootstrap confidence intervals and McNemar's exact test. Five-fold stratified validation partitions the 193 real training patients before fitting any generator. Each training fold independently fits min-max scaling, GAN (500 epochs), cGAN (300 epochs), and VAE (300 epochs); each model generates 500 records. IQR bounds use only the real training fold. The filtered VAE pool is downsampled to the smaller adversarial-pool size, and consensus is recomputed at tolerance 5.0. Classifier scaling is fitted only to the corresponding augmented training fold; SMOTE is fitted within the fold. Validation uses only real patients, and the 83-patient test cohort is excluded throughout. The implementation rejects precomputed pools and stops rather than substituting another method if consensus fails. Generator seeds are 42--46 for folds 1--5; the split and classifier seeds are 42. Settings are fixed from the existing protocol, with no inner hyperparameter search. This is retrospective fold-contained internal validation, not nested model selection or external validation. Python 3.9.6, TensorFlow 2.15.0, NumPy 1.26.4, and scikit-learn 1.5.2 were used for the corrected rerun. Per-fold membership, scaling bounds, generated pools, and AUCs are retained in the accompanying artifact.
```

## 26. Professor comments

Clarify the unperformed competing-risk analysis in third person.

Original:

```tex
{\color{red}Stating that precisely rather than just qualitatively would need a full competing-risks regression, for example a fine-gray model, which no package in our pipeline currently supports, so we did not fit one here.[??NOT CLEAR \& ALWAYS USE 3RD PERSON GRAMMAR]}
```

Replacement:

```tex
A Fine--Gray regression was not fitted; the present analysis therefore does not estimate covariate effects on the cumulative incidence of death in the presence of transplantation.
```

## 27. Professor comments

Include the requested overview of the existing pipeline.

Original:

```tex
{\color{red}[?? CREATE A ABSTRACT DIAGRAM FOR BETTER IMPACT FOR ALL THE PROPOSED ALGORITHM]}
```

Replacement:

```tex
\begin{figure*}[!t]
\revisioncolor
\centering
\includegraphics[width=0.98\textwidth]{fig14_study_overview.png}
\Description{The train and test split precedes generation; fitted pools undergo filtering and consensus before distinct fidelity, disclosure, and prediction evaluations.}
\caption{\revisioncolor Overview of the evaluation. The held-out test partition is separated before generator fitting. Five generative models and a consensus-derived dataset are assessed through distinct fidelity, disclosure-screening, and prediction tasks. The masked-loss VAE uses additional partially observed training records. Generative cross-validation was rerun with independent fitting, filtering, and consensus construction inside each training fold.}
\label{fig:overview}
\end{figure*}
```

## 28. Professor comments

Briefly explain how to read the performance table.

Original:

```tex
{\color{red}Table~\ref{tab:performance} reports complete test-set metrics across all five scenarios.[?? EXPLAIN THIS TABLE A LITTLE BIT MORE BRIEFLY]}
```

Replacement:

```tex
Table~\ref{tab:performance} compares real-only training (A), augmentation (B--D), and synthetic-only training (E), all on the same 83 real test patients. AUC changes depend on the classifier: LR reaches 0.8467 with cGAN augmentation versus 0.8424 at baseline, while synthetic-only cGAN training gives AUC 0.8485 for LR but 0.6867 for GB.
```

## 29. Professor comments

Clarify why one-class VAE landmark results are unavailable.

Original:

```tex
{\color{red}Extending scenario E to all six generators under this framing turns up something the classification framing never showed: the VAE's synthetic pool collapses to a single class after landmark filtering, for every classifier, so we report it as missing rather than a crash (Section~\ref{sec:predutil}). In plain terms, the VAE's generated N\_days and status pairs do not land on both sides of the 3-year mark in this sample, one more sign of the over-smoothing already visible in its FID and KS statistics. The other five generators still work fine, and, as in the main classification framing, show no consistent link between distributional fidelity and landmark AUC.[??NOT VERY CLEAR]}
```

Replacement:

```tex
Under the three-year endpoint, the retained VAE synthetic records contain only one outcome class. Binary classifiers therefore cannot be fitted for that synthetic-only condition, and the corresponding results are reported as unavailable. This identifies a failure to preserve the joint distribution of follow-up duration and event status needed for the specified task. It does not imply that every VAE-based method has this limitation.
```

## 30. Professor comments

Remove the marked unnecessary passage; retain the short inference limitation in Threats to Validity.

Original:

```tex
{\color{red}We did not run a significance test on the C-index, so read this single number the same way as the SMOTE result in the classification framing, a nominal gain we cannot yet confirm, not a contradiction of the null result that holds everywhere else.[??NOT REQUIRED]}
```

Replacement:

```tex
[Removed]
```

## 31. Professor comments

State the augmentation hypothesis in the abstract; use precise local wording here.

Original:

```tex
{\color{red}hypothesis[?? SOMEWHERE IN THE ABSTRACT ASSERT THIS AS HYPOTHESIS]}
```

Replacement:

```tex
relationship between fidelity and task-specific utility
```

## 32. Professor comments

Explain the noise scale and the meaning of the near-duplicate reduction.

Original:

```tex
{\color{red}which output noise at $\sigma=0.7$ brings down to 1.4\% while also improving FID.[??NOT VERY CLEAR]}
```

Replacement:

```tex
which falls to 1.4\% after Gaussian noise with standard deviation $0.7$ times each continuous variable's training standard deviation is added to generated values. This is a distance-based result, not a privacy guarantee.
```

## 33. Technical accuracy

Do not attribute wearable neurological monitoring to EHR studies.

Original:

```tex
Hospitals and clinics that use internet of medical things (IoMT) devices to continuously collect measurements from wearable sensors allows them to monitor neurological disruptions under medically-defined specific health conditions, which are used by decision-support systems to help clinicians detect patient health deterioration
```

Replacement:

```tex
Clinical decision-support research has used electronic health records to identify cirrhosis and predict mortality
```

## 34. Technical accuracy

Remove an unsupported universal deployment claim.

Original:

```tex
An IoMT decision-support system is rarely trained and deployed at the same site: a model developed at one hospital is expected to run at another, on that institution's own patients and its own devices.
```

Replacement:

```tex
Cross-site deployment may require a model developed at one hospital to work on another institution's patients and devices.
```

## 35. Technical accuracy

Do not present all inter-hospital pooling as prohibited.

Original:

```tex
Pooling records across sites would solve the sample-size problem, but it is precisely what data protection law prevents.
```

Replacement:

```tex
Pooling records may increase sample size, but requires appropriate data-governance arrangements.
```

## 36. Technical accuracy

Align the motivation with the actual evaluation.

Original:

```tex
and it is this cross-site question, rather than augmentation in the abstract, that motivates the present study.
```

Replacement:

```tex
although the present study evaluates augmentation and synthetic-only training within one cohort, not transfer between hospitals.
```

## 37. Technical accuracy

Remove an unsupported privacy guarantee.

Original:

```tex
but do not contain any actual patient information
```

Replacement:

```tex
but require separate assessment for memorisation and disclosure risk
```

## 38. Technical accuracy

Synthetic generation alone does not guarantee privacy.

Original:

```tex
as a secure method for sharing healthcare data without risking the exposure of personal patient information
```

Replacement:

```tex
as a possible method for sharing healthcare data while managing disclosure risk
```

## 39. Technical accuracy

Consensus is not a separately trained sixth generator.

Original:

```tex
extended to six generators for the synthetic-only evaluation
```

Replacement:

```tex
extended to six synthetic-data methods (five generators and a derived consensus set) for the synthetic-only evaluation
```

## 40. Technical accuracy

Report the association as an evaluation, not proof of no relationship.

Original:

```tex
and shows across six generators that FID does not significantly predict train-on-synthetic-test-on-real utility.
```

Replacement:

```tex
and examines the association between FID and train-on-synthetic-test-on-real utility across six methods.
```

## 41. Technical accuracy

Avoid claiming removal of all possible leakage.

Original:

```tex
under three framing, leakage-free binary classification
```

Replacement:

```tex
under three framings: binary classification excluding follow-up duration
```

## 42. Technical accuracy

Different outcome analyses do not prove absence of censoring bias.

Original:

```tex
so that the choice to binarise a censored outcome is not itself driving the conclusions.
```

Replacement:

```tex
to examine sensitivity to the outcome definition.
```

## 43. Technical accuracy

Correct a local typographical error.

Original:

```tex
This reminder of the paper
```

Replacement:

```tex
The remainder of the paper
```

## 44. Technical accuracy

Correct local agreement.

Original:

```tex
classic augmentation methods have its own limitations
```

Replacement:

```tex
classic augmentation methods have their own limitations
```

## 45. Technical accuracy

Remove unconditional model-superiority wording.

Original:

```tex
In contrast, deep generative models handle these limitations better, since they learn those relationships directly from the data, and they are now one of the most common ways to generate synthetic clinical tabular data
```

Replacement:

```tex
Deep generative models can learn nonlinear relationships directly from data, although this flexibility does not guarantee better utility or fidelity
```

## 46. Technical accuracy

Repair an incomplete sentence.

Original:

```tex
While VAE uses a different strategy; instead of adversarial training,
```

Replacement:

```tex
A VAE uses a different strategy: instead of adversarial training,
```

## 47. Technical accuracy

Correct punctuation.

Original:

```tex
However good distributional similarity
```

Replacement:

```tex
However, good distributional similarity
```

## 48. Technical accuracy

Distinguish generated columns from predictors and non-laboratory variables.

Original:

```tex
19 relevant features. The 19 features consist of 11 continuous laboratory measurements that includes,
```

Replacement:

```tex
19 variables, including the outcome status. Of these, 11 are continuous variables, including
```

## 49. Technical accuracy

Match both generator-output dimensions to the artifact.

Original:

```tex
sigmoid output of dimension 18
```

Replacement:

```tex
sigmoid output of dimension 19 (18 variables plus status)
```

## 50. Technical accuracy

Loss plots do not establish convergence.

Original:

```tex
The adversarial losses settle without the oscillation that signals mode collapse, and the VAE ELBO falls sharply before levelling off by roughly epoch 50, so all three models are trained to convergence.
```

Replacement:

```tex
The loss curves describe optimisation under the stated epoch budgets; they do not establish convergence or rule out mode collapse.
```

## 51. Technical accuracy

Define IQR and limit the meaning of the filter.

Original:

```tex
A single out-of-range value disqualifies the entire record.
```

Replacement:

```tex
A single out-of-range value disqualifies the entire record. Here $\mathrm{IQR}=Q_3-Q_1$; these statistical bounds do not certify clinical plausibility.
```

## 52. Technical accuracy

Do not portray the selected tolerance as objectively optimal.

Original:

```tex
the point at which class balance becomes acceptable and the quality improvement curve begins to flatten
```

Replacement:

```tex
an empirical trade-off between retained class balance and distributional similarity, not an independently validated optimum
```

## 53. Technical accuracy

Match the implemented FID columns.

Original:

```tex
tabular proxy FID computed on the 18-dimensional feature vectors
```

Replacement:

```tex
tabular proxy FID computed on the 11 continuous variables, including follow-up duration
```

## 54. Technical accuracy

Match the implemented FID columns.

Original:

```tex
over the full 18-dimensional feature space
```

Replacement:

```tex
over the 11 continuous variables
```

## 55. Technical accuracy

Match the implemented FID columns.

Original:

```tex
over the full 18-dimensional feature geometry
```

Replacement:

```tex
over the 11 continuous variables
```

## 56. Technical accuracy

Explain duration leakage without implying all short durations are deaths.

Original:

```tex
because a short N\_days is tied to a recorded death events which is missing for all the patients.
```

Replacement:

```tex
because it includes post-enrolment follow-up information that is unavailable at prediction time.
```

## 57. Technical accuracy

Survival fitting does use censoring indicators.

Original:

```tex
without considering the observability of the event.
```

Replacement:

```tex
with event indicators distinguishing deaths from censored observations.
```

## 58. Technical accuracy

Correct the Aalen--Johansen assumption claim.

Original:

```tex
which does not need a non-informative-censoring assumption.
```

Replacement:

```tex
which accounts for competing events while still requiring appropriate assumptions about other censoring.
```

## 59. Technical accuracy

Frequency alone cannot exclude competing-risk effects.

Original:

```tex
So the competing event is enough that censoring at transplant is unlikely to seriously bias our results.
```

Replacement:

```tex
These proportions alone do not establish that censoring at transplant has negligible impact on interpretation.
```

## 60. Technical accuracy

FID alone does not identify a causal failure mechanism.

Original:

```tex
which confirms that smooth variational generation loses distributional sharpness.
```

Replacement:

```tex
indicating a larger mismatch in feature means and covariances.
```

## 61. Technical accuracy

Distance is not a safety guarantee.

Original:

```tex
fall at a safe distance from any real patient
```

Replacement:

```tex
lie above this empirical distance threshold
```

## 62. Technical accuracy

Avoid an untested causal explanation of proximity.

Original:

```tex
reflecting the broader coverage of the variational latent space and the fact that consensus acceptance pools records lying close to real patients.
```

Replacement:

```tex
indicating greater proximity to training patients under this metric.
```

## 63. Technical accuracy

Distinguish a disclosure indicator from measured attack risk.

Original:

```tex
it is far riskier than the filtered GAN (7.3\%) or cGAN (9.6\%), because it inherits the VAE records that reside closest to real patients.
```

Replacement:

```tex
it has a higher near-duplicate rate than filtered GAN (7.3\%) or cGAN (9.6\%); these rates are not probabilities of re-identification.
```

## 64. Technical accuracy

Separate AUC estimation from paired accuracy inference.

Original:

```tex
but both reside well inside the overlapping bootstrap intervals in Table~\ref{tab:significance} and neither McNemar comparison reaches significance value (Table~\ref{tab:significance}), likely due noise effect rather than a real augmentation effect.
```

Replacement:

```tex
but neither paired accuracy comparison is significant (Table~\ref{tab:significance}). McNemar tests accuracy, not AUC; overlapping individual AUC intervals are not a paired test of their difference.
```

## 65. Technical accuracy

Correct the false all-scenarios AUC claim.

Original:

```tex
every scenario and classifier combination clears AUC~$= 0.8$, so no augmentation strategy damages discrimination, but none uniformly improves on the baseline either.
```

Replacement:

```tex
all classifier combinations in scenarios A--D exceed AUC~$=0.8$, but none of the augmentation strategies uniformly improves on baseline. Synthetic-only GB is lower at 0.6867.
```

## 66. Technical accuracy

Do not infer task difficulty from non-equivalent outcomes.

Original:

```tex
This fits a fixed-horizon task simply being easier to discriminate than an unbounded one; it is not evidence of a stronger augmentation effect.
```

Replacement:

```tex
The endpoint and eligible patient subset differ, so the higher AUCs do not by themselves indicate a stronger augmentation effect.
```

## 67. Technical accuracy

Qualify the censoring and statistical interpretation.

Original:

```tex
augmentation gives no statistically distinguishable improvement, while also avoiding the coarser censoring problem the reviewer raised.
```

Replacement:

```tex
no paired accuracy difference is significant. Excluding patients censored before three years addresses unknown labels but can introduce selection bias.
```

## 68. Technical accuracy

Remove an invalid cross-metric inference.

Original:

```tex
, close to the classification framing's baseline AUC once N\_days is excluded, which tells us that discarding censoring information was not itself hiding a lot of extra signal.
```

Replacement:

```tex
; C-index and classification AUC are different measures and should not be interpreted as interchangeable.
```

## 69. Technical accuracy

Nonsignificance does not demonstrate noise or equivalence.

Original:

```tex
But none of that movement is distinguishable from noise (Table~\ref{tab:significance}).
```

Replacement:

```tex
The paired accuracy comparisons are not significant (Table~\ref{tab:significance}); this does not establish equivalence or test AUC differences.
```

## 70. Technical accuracy

Avoid subgroup significance claims without a corresponding test.

Original:

```tex
So the null result holds across the main clinical groups in this cohort, not just overall.
```

Replacement:

```tex
These descriptive subgroup comparisons do not establish subgroup equivalence.
```

## 71. Technical accuracy

Remove an unsupported qualitative utility ranking.

Original:

```tex
The masked-loss VAE has the best FID (0.062) and a strong mean TSTR AUC (0.798), which fits good utility.
```

Replacement:

```tex
The masked-loss VAE has the lowest FID (0.062) and mean TSTR AUC 0.798.
```

## 72. Technical accuracy

State the limited inference from six method-level observations.

Original:

```tex
No steady relationship survives once a sixth generator joins the comparison. This six-point result is what our narrower abstract claim now rests on, not a guess from a single comparison.
```

Replacement:

```tex
With only six related methods, the association is inconclusive; these results do not establish the absence of a fidelity--utility relationship.
```

## 73. Technical accuracy

A nonsignificant log-rank result does not prove curve equivalence.

Original:

```tex
reproduce the real survival curve closely (log-rank
```

Replacement:

```tex
show no significant log-rank difference from the real curve (
```

## 74. Technical accuracy

Remove the obsolete duration-inclusive importance ranking.

Original:

```tex
N\_days, bilirubin, and albumin, which Fig.~\ref{fig:importance} ranks as the first, second, and fourth most important predictors.
```

Replacement:

```tex
N\_days, bilirubin, and albumin. N\_days is an outcome duration rather than a baseline predictor.
```

## 75. Technical accuracy

Retain the favourable exploratory result without declaring a real effect.

Original:

```tex
The RF gain is the closest thing to a real effect anywhere in this study,
```

Replacement:

```tex
The RF comparison has
```

## 76. Technical accuracy

Separate the positive AUC estimate from its accuracy p-value.

Original:

```tex
but McNemar's test still falls short of significance ($p = 0.125$) and the bootstrap intervals still overlap.
```

Replacement:

```tex
but the paired accuracy comparison is not significant (McNemar $p=0.125$); this is not a test of the AUC gain.
```

## 77. Technical accuracy

Do not apply an accuracy-power calculation to an AUC effect.

Original:

```tex
the proposed test is simply underpowered to confirm an effect this size at $n=83$.
```

Replacement:

```tex
small paired differences are difficult to resolve at $n=83$; the power calculation concerns accuracy, not AUC.
```

## 78. Technical accuracy

Avoid declaring an untested change to be noise.

Original:

```tex
noise-level changes
```

Replacement:

```tex
small descriptive changes
```

## 79. Technical accuracy

Describe observations rather than claiming a new theory.

Original:

```tex
told a different theory
```

Replacement:

```tex
showed a different pattern
```

## 80. Technical accuracy

Correct the FID dimension consistently.

Original:

```tex
a tabular proxy on 18-dimensional feature vectors
```

Replacement:

```tex
a tabular proxy on the 11 continuous variables
```

## 81. Technical accuracy

Do not infer impossibility of differential privacy from a projection.

Original:

```tex
and the proposed DP-SGD analysis shows a formal privacy guarantee is out of reach at this sample size anyway.
```

Replacement:

```tex
and the simplified DP-SGD projections do not establish a formal privacy guarantee.
```

## 82. Technical accuracy

Retain the survival limitations accurately and concisely.

Original:

```tex
Fig.~S11 in the supplementary file suggests this is unlikely to bias the results much, since transplant makes up only 6.5\% of the cohort against 40.2\% for death, but confirming that properly would need a Fine-Gray model, which no package in our pipeline currently supports.
```

Replacement:

```tex
Fig.~S11 describes the competing-event incidence, but does not establish negligible bias. A Fine--Gray regression and paired C-index significance tests were not performed.
```

## 83. Technical accuracy

Correct the comparator model name.

Original:

```tex
so it should not be read as a benchmark of cGAN as originally published.
```

Replacement:

```tex
so it should not be read as a benchmark of CTGAN as originally published.
```

## 84. Technical accuracy

Do not turn relative metrics into a deployment recommendation.

Original:

```tex
the cGAN is the safest choice here: lowest FID paired with a low near-duplicate rate.
```

Replacement:

```tex
the cGAN combines the lowest FID among the three core models with a relatively low near-duplicate rate, but release safety requires a separate assessment.
```

## 85. Technical accuracy

Remove a pattern not shared by every method.

Original:

```tex
and this same pattern showed up for all six generators, not just cGAN.
```

Replacement:

```tex
illustrating why synthetic-only utility must be checked separately for each generator and classifier.
```

## 86. Technical accuracy

Distinguish historical plots from the rerun validation.

Original:

```tex
Every quantity plotted in S6 to S10 also appears numerically in Tables~\ref{tab:performance} and~\ref{tab:cv}.
```

Replacement:

```tex
These historical supplementary plots should not be used as evidence for the corrected fold-contained cross-validation; its results are reported in Table~\ref{tab:cv}.
```

## 87. Technical accuracy

Correct the per-feature ranking overstatement.

Original:

```tex
Cohen's $d$ ranks the VAE as the best-matched method on every feature even though it is the worst on every distributional-shape measure, which is a further instance of the paper's central point that a single fidelity metric can be satisfied by a model that is wrong in every other respect.
```

Replacement:

```tex
the VAE has small absolute standardised mean differences despite substantial distributional-shape differences. It is not the closest method on every individual mean-difference measure, reinforcing the need for complementary quality criteria.
```

## 88. Validation correction

Insert the verified fold-contained validation interpretation.

Original:

```tex
Table~\ref{tab:cv} presents 5-fold cross-validation AUC results, computed with synthetic records confined to the training side of each fold and every validation fold containing real patients data only, on the leak-free 17-feature input. Under this protocol, neither deep-generative augmentation scenario (B or C) improves the mean cross-validated AUC over the real-data baseline for RF or GB; both lower the mean for these two classifiers while marginally raising it for LR (0.8357 and 0.8349 against a baseline of 0.8321). SMOTE (scenario D) is a partial exception, worth stating plainly rather than folding into the same null result as B and C: it raises the RF mean to 0.8559, above the baseline's 0.8499, while lowering GB to 0.8126 and leaving LR essentially unchanged (0.8300). This is the same pattern the reviewer identified in the pre-correction numbers, now confirmed on the leak-free feature set: a classical interpolation baseline can nominally beat real-data-only training on a subset of classifiers, even though no deep-generative scenario does. The fold-to-fold standard deviation shows no consistent direction either: GB falls modestly from 0.0535 (scenario A) to 0.0480 (scenario C), while RF rises from 0.0464 to 0.0532 over the same comparison. This is a different picture from an earlier, since-corrected version of this analysis, in which allowing synthetic records into the validation folds produced a large apparent variance reduction; no such effect appears once validation is restricted to real patients, with or without N\_days as a feature.
```

Replacement:

```tex
Table~\ref{tab:cv} replaces the earlier generative cross-validation estimates with a completed fold-contained rerun. Each of the five folds retrains GAN, cGAN, and VAE using only its 154 or 155 real training patients, generates 500 records per model, and recomputes IQR filtering, equalisation, and consensus. Validation uses the remaining 38 or 39 real patients. Both generative augmentation strategies have lower mean AUC than the real-only baseline for all three classifiers. For consensus, LR mean AUC is 0.7803 rather than the earlier, fold-leaked estimate of 0.8349. SMOTE yields a small descriptive RF increase from 0.8499 to 0.8559. Means and standard deviations summarise five correlated folds; these comparisons are not significance tests or confidence intervals.
```

## 89. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.8476 \pm 0.0537$
```

Replacement:

```tex
$0.8227 \pm 0.0573$
```

## 90. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.8169 \pm 0.0586$
```

Replacement:

```tex
$0.8081 \pm 0.0688$
```

## 91. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.8357 \pm 0.0502$
```

Replacement:

```tex
$0.8184 \pm 0.0618$
```

## 92. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.8295 \pm 0.0532$
```

Replacement:

```tex
$0.8377 \pm 0.0723$
```

## 93. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.7880 \pm 0.0480$
```

Replacement:

```tex
$0.8088 \pm 0.0573$
```

## 94. Validation correction

Replace an invalid generative CV estimate with its verified rerun mean and fold SD.

Original:

```tex
$0.8349 \pm 0.0523$
```

Replacement:

```tex
$0.7803 \pm 0.0404$
```

## 95. Validation correction

Specify corrected validation and avoid interpreting SD as a confidence interval.

Original:

```tex
Mean $\pm$ Std, real-only validation folds, N\_Days excluded
```

Replacement:

```tex
Mean $\pm$ Fold SD, fold-contained generation, real-only validation, N\_Days excluded
```

## 96. Validation correction

Update Discussion to agree with the corrected CV values.

Original:

```tex
The cross-validation results back this up rather than overturn it. Repeating training across five folds, with synthetic data kept only on the training side, shows no augmentation scenario improving the mean AUC over baseline for RF or GB. The fold-to-fold variance moves in no consistent direction either (GB goes from 0.0535 to 0.0480 between scenario A and C; RF goes the other way, from 0.0464 to 0.0532). An earlier version of this analysis let synthetic records into the validation folds and appeared to show a large drop in variance, but that was just an artefact: consensus records are drawn from the dense centre of feature space, so they are easier and more uniform to predict, and including them in validation mechanically shrinks the measured variance. Once validation is restricted to real patients, that apparent stability gain disappears, whether or not N\_days is included. This reinforce the proposed argument that never evaluate synthetic augmentation on folds that themselves contain synthetic data.
```

Replacement:

```tex
The corrected cross-validation results do not support an augmentation benefit from cGAN or consensus: their mean AUCs are below the real-only baseline for all three classifiers. The small RF mean-AUC increase with within-fold SMOTE is descriptive, not a formal paired test of improvement. These results reinforce the distinction between synthetic-data resemblance and useful augmentation.
```

## 97. Technical accuracy

Correct the false claim that DP-SGD models were trained; keep the existing perturbation results.

Original:

```tex
The full VAE privacy-enhancement sweep is plotted in Fig.~S15 of the supplementary file (refer Tables~S6 and~S7). This information is relevant because consensus voting alone does not give a low-risk set either (21.1\% near-duplicates, Table~\ref{tab:privacy}). The filtered VAE pool of 499 records has a 35.7\% near-duplicate rate, so more than one in three generated records resides closer to a real training patient than the 5th-percentile real-to-real distance. To fix this without retraining the model, Gaussian noise has been added to each VAE record's eleven continuous features after generation, scaled to each feature's training-data spread. At $\sigma=0.7$, the near-duplicate rate drops from 35.7\% to 1.4\%, and FID actually improves from 0.228 to 0.0481, better than the equalised consensus FID of 0.0809. That FID improvement makes sense in a way that the VAE was clustering records too close to specific real patients, so spreading those clusters out brings its overall distribution closer to the real one. The formal DP-SGD is also trained across noise multipliers from 0.5 to 3.0, giving epsilon values between 14.0 and 210.4 at $\delta=10^{-5}$, far too large for any meaningful privacy guarantee. So output noise at $\sigma=0.7$ is the most practical fix at $n=193$. Formal differential privacy through DP-SGD only becomes realistic once a cohort reaches several hundred patients.
```

Replacement:

```tex
The output-perturbation sweep is shown in supplementary Fig.~S15 and Table~S6. Gaussian noise is added independently to each of the 11 continuous VAE variables, with standard deviation $\sigma s_j$, where $s_j$ is the corresponding real-training standard deviation; negative perturbed values are clipped to zero. At $\sigma=0.7$, the reported near-duplicate rate falls from 35.7\% to 1.4\%, while the tabular FID falls from 0.228 to 0.0481. This indicates a change in proximity and moment matching, not demonstrated protection against an adversary. Downstream utility and clinical validity after perturbation require separate assessment. The DP-SGD values in supplementary Table~S7 are approximate analytical projections from a simplified privacy-accounting calculation, not results from differentially private model training or certified privacy guarantees.
```

## 98. Technical accuracy

Keep the theory subsection and augmentation rationale, but remove the unsupported impossibility theorem.

Original:

```tex
A generative model trained on the 193 real training patients is just a function of those patients plus some randomness. Whatever it produces is a transformation of information already existing in the training set. By the data processing inequality, synthetic records cannot carry more information about the true population than the data they came from. Augmentation can reweight or smooth what the training data already shows, but it cannot add fresh evidence about patients the model never saw. That puts a hard ceiling on what augmentation can achieve, and on a cohort where the leak-free baseline classifier already reaches AUC~$=0.842$, there is not much headroom left beneath that ceiling, though more than the earlier analysis suggested when N\_days had inflated the apparent baseline to 0.878.
```

Replacement:

```tex
A generative model trained on the 193 real training patients is a function of those patients plus randomness. Its synthetic records do not add independent patient observations. Augmentation can nevertheless reweight or smooth the training distribution, regularise a classifier, or change class balance. The theory motivating this study is therefore that these changes may improve prediction; the empirical question is whether the proposed filtering and consensus procedure produces such a benefit on held-out real patients.
```

## 99. Technical accuracy

Retain the proposed mechanism as a hypothesis rather than an established cause.

Original:

```tex
The consensus mechanism actually works against itself here. Cross-architecture agreement, by design, only accepts records that all three models place in the same region of feature space, and that region is the dense centre of the training distribution. The simulated privacy results confirm this directly: the consensus set has a 21.1\% near-duplicate rate, higher than either adversarial pool, meaning its records sit unusually close to real patients. Records that close to existing training points are, almost by definition, redundant to a classifier. They reinforce parts of the feature space that are already well covered and add nothing near the decision boundary, the only place extra samples could actually change a fitted model.
```

Replacement:

```tex
Cross-architecture agreement, by design, favours records represented in all three generated pools. The consensus set has a 21.1\% near-duplicate rate, higher than either adversarial pool, suggesting that proximity and diversity warrant attention. Concentration in already represented regions is one possible explanation for the limited augmentation benefit, but these experiments do not establish that every accepted record is redundant or that only decision-boundary samples can improve learning.
```

## 100. Technical accuracy

Identify the classifier mechanism as an interpretation.

Original:

```tex
This also explains why the effect differs by classifier.
```

Replacement:

```tex
Classifier sensitivity is another possible explanation for the differing effects.
```

## 101. Technical accuracy

Tree models can change even when records fall within existing partitions.

Original:

```tex
so points that fall inside an existing partition leave the split thresholds almost untouched.
```

Replacement:

```tex
and additional training points may alter split thresholds or leaf estimates.
```

## 102. Technical accuracy

Remove the false largest-movement claim and unsupported causality.

Original:

```tex
which is why it shows the largest movement across scenarios in Table~\ref{tab:performance}.
```

Replacement:

```tex
although the present experiments do not isolate this mechanism.
```

## 103. Technical accuracy

Avoid generalising one cohort into a universal augmentation limit.

Original:

```tex
Synthetic data can substitute for real data that is missing, but it cannot add to real data that is already enough. That distinction, substitution rather than augmentation, is the practical lesson of this study.
```

Replacement:

```tex
Synthetic-only training therefore retains predictive signal in this split, while augmentation shows no consistent advantage. This does not establish equivalence to real data or a sample-size threshold beyond which augmentation cannot help.
```

## 104. Technical accuracy

Keep the IoMT motivation but remove claims of experiments and guarantees not performed.

Original:

```tex
Scenario~E matters most for IoMT settings where hospitals cannot pool patient data. GDPR and HIPAA stop a hospital from exporting identifiable records, so a model that needs data from several sites cannot simply be trained on all of them together. The proposed results show the records themselves may not need to move. One site can train a generator locally and release only synthetic records. A second site can then train a classifier on that output and still reach AUC 0.81 to 0.85 on its own real patients, without ever seeing a real record from the first site. This keeps disclosure risk negligible: the filtered cGAN pool has a 9.6\% near-duplicate rate, and output perturbation brings the VAE's rate down from 35.7\% to 1.4\% where a stricter guarantee is needed. This approach also suits IoMT infrastructure well: it moves less data over constrained uplinks, lets the receiving site keep and reuse what it trains on, and reduces privacy governance to a one-time check on the synthetic set rather than ongoing per-record consent.
```

Replacement:

```tex
Scenario~E is relevant to possible IoMT settings where access to real patient records is restricted. The results show that classifiers trained on synthetic cGAN records can retain predictive signal on held-out patients from the same source cohort. They do not demonstrate transfer between hospitals, reduced communication costs, or a compliant release mechanism. The cGAN near-duplicate rate of 9.6\% and the perturbed VAE rate of 1.4\% are empirical distance indicators, not negligible-risk or formal privacy guarantees. Cross-site use remains a motivation for future evaluation rather than a deployment validated in this study.
```

## 105. Technical accuracy

Replace unsupported cross-generator and cross-site generalisations.

Original:

```tex
Two limits are worth stating here. First, GB fell to AUC~$0.687$ under scenario~E, and this same weakness shows up for every generator that has been tested in this work (mean TSTR AUC 0.777 to 0.822 across all six), so a receiving site must check its own choice of model rather than assume this transfers safely. Second, synthetic transfer is only worth doing where real data genuinely cannot be shared. Where pooling is allowed, Table~\ref{tab:performance} shows real data should just be used directly, since augmentation adds nothing once enough real records are already available.
```

Replacement:

```tex
Two limits are worth stating here. First, synthetic-only cGAN training gave AUC 0.687 for GB, compared with 0.809 for RF and 0.849 for LR, so usefulness is classifier dependent. Second, the results come from one retrospective cohort; a receiving institution would need its own utility, privacy, and governance assessment before deployment.
```

## 106. Validation correction

Make practical guidance match the corrected leakage mechanism.

Original:

```tex
The study conducted in this paper suggests four crucial aspects that should be followed for building clinical decision support on a small tabular cohort like this one. First, refrain from evaluating augmentation on validation folds that contain synthetic records. In this study, the earlier variance reduction disappeared completely once we restricted validation to real patients, so treat any stability gain measured on mixed folds as suspect until it survives a real-only check.
```

Replacement:

```tex
The study suggests four practical considerations for small tabular cohorts. First, validation must contain real patients only, and every fitted preprocessing, generation, filtering, and selection step must use only the corresponding training fold. Restricting validation to real records is insufficient if the generator has already seen those patients.
```

## 107. Technical accuracy

Clarify reproducibility scope and seed differences without claiming every analysis was rerun.

Original:

```tex
The Mayo Clinic PBC dataset is publicly available from the UCI Machine Learning Repository \cite{ucipbc}. All generative models, filtering and consensus code, and evaluation scripts are implemented in Python and TensorFlow~2.15 and run under fixed random seeds (Python, NumPy, and TensorFlow all seeded at 42), so the full pipeline, including synthetic data generation, reproduces the numbers reported here when re-executed in the same software environment. The survival-modelling framing additionally uses \texttt{lifelines} for Cox proportional-hazards fitting and, where available, \texttt{scikit-survival} for the Random Survival Forest. Because TensorFlow does not guarantee bitwise-identical arithmetic across different hardware or thread configurations, small numerical differences are possible on other systems. The code is available at https://github.com/Michaeludousoro/liver-cirrhosis-synthetic-data.
```

Replacement:

```tex
The Mayo Clinic PBC dataset is publicly available from the UCI Machine Learning Repository \cite{ucipbc}. The project provides generative-model, filtering, consensus, and evaluation scripts in Python, with TensorFlow 2.15 used for the core neural generators. The fixed split and classifiers use seed 42; the corrected generator rerun uses seeds 42--46 across its five folds. Survival analyses use \texttt{lifelines} and \texttt{scikit-survival}. Reproduction requires the corresponding script sequence and software environment; fixed seeds do not ensure identical results across hardware or thread configurations. Code: \url{https://github.com/Michaeludousoro/liver-cirrhosis-synthetic-data}.
```

## 108. Technical accuracy

Describe assistance without claiming unverified coauthor approval.

Original:

```tex
The authors used a generative AI assistant for code review, debugging, language editing, and, in response to peer review, implementing the corrected leak-free feature set, the landmark and survival-modelling framings, the six-generator extension of Scenario E, and the resulting pipeline reruns and analysis code described in this paper. All experiments, results, and scientific conclusions were verified by the authors against the underlying code and data, who take full responsibility for the content of this paper.
```

Replacement:

```tex
The authors used a generative AI assistant for code review, debugging, language editing, and assistance with the evaluation scripts described in this paper. The present editorial revision also used AI assistance for literature discovery, reporting corrections, and preparation of figure code. The authors retain responsibility for verifying the experiments, references, and scientific conclusions before submission.
```

## 109. Technical accuracy

Overlapping AUC intervals are not the basis of McNemar accuracy inference.

Original:

```tex
all confidence intervals overlap substantially across scenarios, including SMOTE, and all nine McNemar p-values exceed 0.06, confirming that
```

Replacement:

```tex
all nine McNemar p-values exceed 0.06, so
```

## 110. Technical accuracy

Do not present the two closest p-values as promising gains.

Original:

```tex
both are short of $\alpha = 0.05$ and, per Section~\ref{sec:power}, the test is underpowered at this sample size regardless.
```

Replacement:

```tex
both exceed $\alpha=0.05$ and both comparisons have more newly incorrect than newly correct predictions. The small test set limits inference.
```

## 111. Technical accuracy

Do not infer overall superiority from fidelity alone.

Original:

```tex
The simpler model still outperforms;
```

Replacement:

```tex
Under this training budget, the simpler model has lower FID;
```

## 112. Technical accuracy

Identify the convergence explanation as tentative.

Original:

```tex
This gap looks like a property of the small cohort rather than a flaw in CTGAN itself:
```

Replacement:

```tex
The training budget may contribute to this gap:
```

## 113. Technical accuracy

Do not assert unmeasured convergence status.

Original:

```tex
This is likely short of full convergence at this cohort size
```

Replacement:

```tex
Convergence was not established at this training budget
```

## 114. Technical accuracy

Use a verifiable training-budget description.

Original:

```tex
a partially-converged snapshot
```

Replacement:

```tex
a fixed-budget snapshot
```

## 115. Technical accuracy

Protocol-related missingness does not establish absence of selection bias.

Original:

```tex
so this exclusion follows trial protocol, not anything about the patients themselves.
```

Replacement:

```tex
so much of this missingness reflects protocol differences; complete-case selection can still introduce bias.
```

## 116. Technical accuracy

Name the consensus-derived comparison correctly.

Original:

```tex
across the six generators
```

Replacement:

```tex
across the six synthetic-data methods
```

## 117. Technical accuracy

Name the consensus-derived comparison correctly.

Original:

```tex
all six generators
```

Replacement:

```tex
all six synthetic-data methods
```

## 118. Technical accuracy

Name the consensus-derived comparison correctly.

Original:

```tex
Across Six Generators
```

Replacement:

```tex
Across Six Synthetic-Data Methods
```

## 119. Technical accuracy

Consensus is a derived set, not an independent generator.

Original:

```tex
as a sixth generator
```

Replacement:

```tex
as an additional generator alongside the consensus-derived set
```

## 120. Technical accuracy

Distinguish historical generation seeds from corrected fold seeds.

Original:

```tex
same seed-42 setup as every other generator
```

Replacement:

```tex
seed-42 setup used for the original full-training-pool generators
```

## 121. Technical accuracy

Preserve the promising partial-data finding without claiming that it reflects a proven effect.

Original:

```tex
No scenario in this section overturns the paper's main null hypothesis result at conventional significance. But the masked-loss VAE result is the one place where the size and direction of the effect, not just its sign, plausibly reflects something real: a modest, classifier-specific benefit from using more of the real signal we already have, through masked training rather than throwing partial records away.
```

Replacement:

```tex
The masked-loss VAE results motivate further investigation of partially observed records. The favourable RF AUC estimate is exploratory, and the non-significant paired accuracy test neither confirms a benefit nor rules one out.
```

## 122. Technical accuracy

Qualify the interpretation of the historical power calculation.

Original:

```tex
This does not undo the null hypothesis.
```

Replacement:

```tex
This conditional calculation does not establish whether a true effect exists.
```

## 123. Technical accuracy

Avoid implying exact agreement of rounded discordance summaries.

Original:

```tex
a mean of 8 pairs, a 9.2\% discordant rate on $n=83$
```

Replacement:

```tex
approximately eight discordant pairs, or roughly 9\% of $n=83$
```

## 124. Technical accuracy

Repair the incomplete related-work transition without introducing a new concept.

Original:

```tex
The main question about following the path from distributional fidelity to predictive utility has received much less attention, especially for small tabular clinical datasets used in IoMT environments. \cite{kababji2023}, \cite{espinosa2023}, and that is the main motivation of this study to test predictive utility on unseen real patients, not just similarity metrics.
```

Replacement:

```tex
These findings motivate testing predictive utility on unseen real patients alongside distributional similarity, particularly for small clinical datasets \cite{kababji2023,espinosa2023}.
```

## 125. Technical accuracy

Repair the incomplete masked-loss methods sentence.

Original:

```tex
The trained VAE whose reconstruction loss only counts
```

Replacement:

```tex
A VAE was trained with a reconstruction loss that only counts
```

## 126. Technical accuracy

Distinguish censored status from known survival.

Original:

```tex
The death has been encoded as event (1) and both staying alive and getting a transplant as non-events (0).
```

Replacement:

```tex
Death is encoded as event (1), while censoring and transplantation are encoded as non-events (0).
```

## 127. Professor comments

Replace the repetitive conclusion with a 350--400-word summary of the existing study, findings and limitations.

Original:

```tex
{\color{red}[??CONCLUSION NEED TO BE BRIEF AND PRECISE] This paper set out to find what synthetic patient data can and cannot do for predicting liver cirrhosis survival in an IoMT setting. During peer review of an earlier version, we found that N\_days, the follow-up duration used to work out Status, had been used as a classifier input. Since it leaks outcome information, this had inflated the reported baseline AUC from 0.842 to 0.878. Every result here now uses a 17-feature leak-free input. We also test the problem two further ways that avoid the cruder step of simply discarding censoring: fixed-horizon landmark classification, and full Cox/Random-Survival-Forest time-to-event modelling. We also extended the synthetic-only (Scenario E) evaluation from one generator to \rev{all six synthetic-data methods} available in this study.

The corrected picture is more nuanced than our original one. With the leak removed, the real-data baseline is weaker than first reported, which reopens some real headroom. The two core deep-generative augmentation scenarios still cannot close that gap at statistical significance under any framing we tested (classification, landmark, or survival). But a masked-loss VAE trained on 335 partially observed patients, rather than the usual 193 complete cases, comes closest to a real effect in this study (Random Forest AUC $+0.035$, $p=0.125$). This points to recovering discarded partial records as a more promising direction than any of the generative-augmentation strategies we tested. Across \rev{all six synthetic-data methods}, FID does not reliably predict train-on-synthetic-test-on-real utility (Pearson $r=0.35$, $p=0.50$; Spearman $\rho=0.26$, $p=0.62$): the masked-loss VAE pairs the best FID with strong utility, but the Vanilla GAN reaches the highest utility from a middling FID, and the VAE reaches competitive utility from the worst FID of any generator we tested. Synthetic data substitutes for real data that is unavailable, rather than adding to real data that is already enough. That distinction, together with the evidence that distributional fidelity does not predict utility, is what this study contributes.

We trained three generative models from scratch, under fixed random seeds, on the Mayo Clinic PBC dataset, plus a masked-loss VAE trained on a larger partially observed pool and a full CTGAN implementation. An IQR filter discarded roughly 34\% to 59\% of generated records \rev{across the six synthetic-data methods} for being clinically implausible. A consensus voting step then kept only records that all three core architectures agreed on, giving an equalised set of 787 records.

Our main findings are these. The masked-loss VAE gave the best distributional quality, FID~$=0.062$, just ahead of cGAN at 0.072, and the filtered adversarial pools carried the lowest disclosure risk (7.3\% near-duplicates for GAN, 9.6\% for cGAN). No augmentation scenario beat the leak-free real-data baseline by a significant margin on the 83-patient test set, under any of the three framings. SMOTE nominally beat the baseline's Random Forest cross-validation mean (0.856 versus 0.850) but not by a significant margin, matching exactly the pattern the reviewer flagged in the pre-correction analysis. Consensus voting did not lower disclosure risk either, staying at 21.1\% near-duplicates, and its synthetic survival curve differs significantly from the real one ($p=0.010$ by log-rank test) despite a mid-range FID, one more sign that FID alone does not capture everything about utility. Classifiers trained only on synthetic cGAN data, with no real patients at all, still reached AUC 0.81 to 0.85 for Random Forest and Logistic Regression on real held-out patients (Gradient Boosting fell to 0.69, a pattern that held across \rev{all six synthetic-data methods}). This confirms synthetic data carries real, though model-dependent, predictive signal, useful for federated IoMT settings where patient data cannot cross sites.

Three limitations remain. McNemar's test is underpowered at $n=83$: roughly 270 held-out patients would be needed to detect a 5-point accuracy gain with 80\% power at the discordant-pair rate we actually observed, so a small real benefit cannot be ruled out. VAE near-duplicate risk falls from 35.7\% to 1.4\% with output noise at $\sigma=0.7$, but formal differential privacy through DP-SGD is not workable until the cohort grows to several hundred patients. The SMOTE-style method we used for the survival framing, interpolating duration together with covariates for the event class, is a documented departure from off-the-shelf SMOTE, not a separately validated method. Replicating this on a larger, multi-site hepatology dataset, and extending it to ongoing IoMT data streams, are the priorities for future work.}


```

Replacement:

```tex
\rev{This paper evaluated what synthetic patient data can contribute to predicting liver cirrhosis outcomes using the Mayo Clinic PBC dataset. Of 418 records, 276 complete cases provided 193 training patients and 83 held-out test patients. Three core generative models---Vanilla GAN, cGAN, and VAE---were combined with IQR filtering and consensus voting. Additional comparisons included SMOTE, a masked-loss VAE using partially observed records, and CTGAN. Follow-up duration was excluded from the 17 baseline predictors, while landmark classification and survival modelling examined alternative outcome definitions.}

\rev{The results distinguish distributional fidelity from predictive utility. The masked-loss VAE achieved the lowest tabular FID, 0.062, followed by cGAN at 0.072. However, cGAN and consensus augmentation did not consistently improve held-out discrimination. Corrected five-fold validation, with generators and filtering fitted independently within each training fold, produced lower mean AUCs for both strategies than real-only training across all three classifiers. SMOTE gave a small descriptive Random Forest increase, from 0.8499 to 0.8559. These fold summaries are not significance tests.}

\rev{The masked-loss VAE produced a favourable exploratory Random Forest AUC change, from 0.832 to 0.868, while the paired accuracy comparison remained non-significant (McNemar $p=0.125$). Synthetic-only cGAN training retained predictive signal, reaching AUC 0.809 for Random Forest and 0.849 for Logistic Regression, compared with 0.687 for Gradient Boosting. These findings support classifier-specific evaluation, not equivalence to real-data training. The six-method fidelity--utility association was inconclusive, and survival C-index differences were descriptive rather than formally tested.}

\rev{Consensus filtering retained 787 records but did not reduce the near-duplicate indicator relative to the adversarial pools: its rate was 21.1\%, compared with 7.3\% for GAN and 9.6\% for cGAN. Output perturbation reduced the VAE indicator, but neither these distances nor the simplified DP-SGD projections establish a privacy guarantee. Statistical filtering should therefore complement, rather than replace, task-specific utility and disclosure assessment.}

\rev{The principal limitations are the small single-cohort test set, complete-case selection, retrospective internal validation, and limited random-seed evaluation. Non-significant accuracy comparisons do not rule out modest benefits or harms. Recovering partially observed records remains a promising direction for further investigation. Larger external cohorts, repeated-seed validation, and direct cross-site evaluation are needed before drawing deployment conclusions for IoMT applications.}


```

## 128. Professor comments

Repair the grammar of the marked introductory transition.

Original:

```tex
\rev{Developing and evaluating these systems}, they require large and well-labelled datasets, and obtaining such datasets is difficult.
```

Replacement:

```tex
Developing and evaluating these systems requires large and well-labelled datasets, which can be difficult to obtain.
```

## 129. Technical accuracy

Make the stated p-value range match the unchanged significance table.

Original:

```tex
from 0.070 to 1.00
```

Replacement:

```tex
from 0.063 to 1.00
```

## 130. Technical accuracy

Distinguish lack of a consistent benefit from absence of any predictive signal.

Original:

```tex
and still gave no predictive benefit.
```

Replacement:

```tex
and still gave no consistent augmentation benefit.
```

## 131. Technical accuracy

Use the recorded outcome label rather than imply known long-term survival.

Original:

```tex
training: 115 survived/transplanted
```

Replacement:

```tex
training: 115 censored/transplanted
```
