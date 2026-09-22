# Targeted correction record — 17 September 2026

This draft supersedes the earlier, more extensively rewritten ACM draft for coauthor review. It is based on the manuscript in the supplied IEEE_Synthetic Data_latex.zip. The earlier files and original manuscript have not been overwritten.

## Scope retained

The title remains exactly: **Theory of Augmentation for Improved Prediction Accuracy: Evaluating the Predictive Utility of Synthetic Liver Cirrhosis Data with Consensus-Based Quality Filtering**.

The original research question, cohort, model architectures, scenarios A–E, theory subsection, and main section sequence remain. No new disease, dataset, research objective, experiment, or deployment study was introduced in this targeted revision. Existing supplementary analyses remain supporting analyses, not new work performed during this edit. The supplementary title has also been brought into agreement.

The five keywords are: synthetic data generation; liver cirrhosis; predictive utility; consensus voting; patient privacy.

## Responses to the professors’ recorded comments

1. **Brief abstract, problem, method, hypothesis and conclusion:** shortened to approximately 230–240 words, depending on treatment of mathematical notation and hyphenated words. It states the augmentation hypothesis, the cohort and procedure, selected findings, and the practical conclusion. It does not present an accuracy improvement as proven.
2. **Brief, precise conclusion:** replaced the repetitive conclusion with approximately 370 words, within the requested 350–400 range. It summarises the existing study rather than adding new arguments.
3. **Marked introduction wording and struck-through phrases:** rephrased the marked transition and removed the struck-through repetition. Most of the original introduction remains.
4. **Avoid an unnecessarily negative account of the contribution:** the related-work paragraph now explains what the evaluation establishes about task-specific quality assessment. Unfavourable results have not been hidden or reversed.
5. **Related-work comparison:** retained the requested comparison table. Relevant additions from JDIQ and ACM Transactions on Computing for Healthcare are included because of their methodological relevance, not simply to cite a prospective venue. The JDIQ paper concerns time-series quality assessment; the HEALTH paper concerns neonatal augmentation and filtering, not liver cirrhosis.
6. **Raincloud plot:** retained the training-only plot of selected biomarkers, with explanatory text. Held-out patients do not determine the plot’s training-data filtering thresholds.
7. **Algorithm overview:** retained the requested pipeline diagram showing the existing study design.
8. **Missing equation definitions:** defined the GAN and tabular FID symbols.
9. **cGAN and VAE naming:** clarified that the implemented cGAN is not CTGAN, and the implemented VAE is not TVAE. The original architecture detail remains.
10. **Tukey filter reference:** added the NIST reference, defined IQR, and distinguished statistical filtering from clinical validation.
11. **Consensus explanation:** added a concrete example of a candidate requiring neighbours from the other two model pools. This is agreement in feature space, not identification of the same patient.
12. **StandardScaler and quality metrics:** provided the standardisation equation and software reference; briefly explained the complementary distributional metrics and privacy-distance scaling.
13. **Follow-up duration:** clarified why N_Days is generated for outcome analyses but excluded from the 17 baseline predictors.
14. **“Why two baselines?”:** Scenario A is the real-only reference; Scenario D is the classical oversampling comparator.
15. **Python version and reproducibility:** supplied the corrected validation environment and seed settings. The short seed-42 note explains that 42 is arbitrary, not an accuracy-tuning choice.
16. **Competing-risk paragraph and third-person wording:** clarified what was and was not estimated. No new competing-risk regression is claimed.
17. **Performance table explanation:** added a short guide to scenarios A–E and examples of classifier-dependent results; the verified held-out values are unchanged.
18. **Single-class landmark VAE result:** explained that binary fitting is unavailable when the retained synthetic pool has only one outcome class.
19. **Unnecessary survival-results passage:** removed the marked passage, retaining a short statement of the unperformed C-index significance testing in the limitations.
20. **Privacy-noise explanation:** specified what the noise scale means and distinguished the reduced near-duplicate indicator from a privacy guarantee.

## Necessary technical corrections — not additional professor requests

These changes prevent the manuscript from describing something different from the implemented study. They do not expand its research scope.

- **Validation leakage:** the old cGAN and consensus cross-validation reused synthetic pools fitted before the folds were separated. Keeping synthetic rows out of validation alone did not remove this indirect leakage. The completed earlier correction retrained the generators and fitted preprocessing, filtering and consensus within each training fold. The targeted draft uses those verified results; no further experiments were run for this edit.
- **Corrected CV results:** cGAN mean AUCs are 0.8227, 0.8081 and 0.8184 for RF, GB and LR; consensus values are 0.8377, 0.8088 and 0.7803. Baseline and SMOTE values are unchanged. The table reports fold standard deviations, not confidence intervals or formal tests of improvement.
- **Validation scope:** this is retrospective internal validation with fixed existing settings, not nested hyperparameter selection, repeated-seed validation or external validation. The split and classifier seeds are 42; generator seeds for the corrected five folds are 42–46.
- **Dimensions:** the core generators output 19 columns including status; classifiers use 17 baseline predictors; FID uses 11 continuous variables; privacy distance uses 18 non-status variables; consensus distance uses all 19 generated columns.
- **Statistical interpretation:** McNemar tests paired classification accuracy, not AUC. Overlap of separate AUC intervals is not a paired AUC test. Non-significant findings are not proof of equivalence. Survival differences and small subgroup comparisons remain descriptive.
- **Privacy and deployment:** near-duplicate rates are distance-based indicators, not re-identification probabilities or safety guarantees. DP-SGD values are simplified projections, not results of trained private models. The study did not test transfer between hospitals or an IoMT deployment.
- **Theory and title:** the original theory subsection and requested title are retained. The argument now explains why augmentation could help or harm; it does not claim a theorem that improvement is impossible. The results test the title’s motivation but do not establish a general accuracy gain.
- **Consistency:** corrected the obsolete feature-importance wording, the false claim that every scenario exceeds AUC 0.8, and references to “six generators” where the sixth method is a derived consensus set.

## What was verified and what was not rerun

The previously completed artifact work reproduced the saved-pool classification results and FID scores, checked the masked-VAE results, and completed the fold-contained generator validation with provenance checks. The targeted revision reuses this evidence. It does not claim fresh retraining of the historical landmark, survival, subgroup, privacy-sweep or partial-record-control analyses.

The full project evidence remains in revisions/2026-09-17/validation. Selected summary evidence is also included with this targeted package. The editable source has one shared manuscript body for the blue and clean copies.

## How to read the two copies

Blue marks corrected or added wording and changed values relative to the supplied manuscript. Unchanged wording remains black. Newly requested comparison and explanatory figures have blue captions; images retain their original colours. Removed text is documented here and in the detailed change ledger rather than shown as blue deletions. ACM layout changes are not themselves scientific changes.

The clean copy has the same manuscript text without revision colours or the reviewer-copy notice. The supplementary document retains the earlier necessary technical corrections; this pass only aligns its title. Its broader marking is not a new targeted rewrite.

## Suggested explanation to coauthors

“I have kept the agreed title and the original study scope. This version addresses the marked comments and shortens the abstract and conclusion, with corrections shown in blue. I have also retained the necessary validation and reporting corrections so that the claims agree with the artifact. The results assess whether augmentation improves prediction; they do not demonstrate a consistent improvement. I will raise any alternative title separately for your agreement.”

The title has not been silently replaced to fit a venue. A more neutral title could be discussed separately because this is an empirical evaluation with mixed outcomes, not a proven general theory of improved accuracy. Final venue selection and coauthor approval remain author decisions. No email has been sent.
