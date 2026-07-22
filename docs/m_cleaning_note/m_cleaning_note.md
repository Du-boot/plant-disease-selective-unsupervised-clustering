# M_new_drop5_drop7 quality-control note

This note is intentionally conservative. The cleaned M dataset should not be
justified by improved clustering results alone.

Available label_map: `/data1/D/M_new_drop5_drop7/label_map.csv`

Retained old_label values found in label_map:

0, 1, 2, 3, 4, 5, 6, 8, 9

Recommended manuscript wording:

> The quality-controlled M subset is reported as the main stress-test set,
> while the complete M_new set is retained as a robustness comparison. The
> exclusion criteria must be based on dataset-level quality-control criteria
> such as ambiguous class definition, non-leaf disease images, label noise,
> corrupted files, insufficient samples, or semantic duplication, rather than
> model performance.

Important limitation:

> Compared with complete M_new, the cleaned subset improves post-hoc aligned
> retained-sample accuracy and coverage, but some structure metrics such as
> ARI/NMI do not improve monotonically. Therefore the cleaned set should be
> described as a quality-controlled subset, not as evidence that clustering
> quality improved in every metric.
