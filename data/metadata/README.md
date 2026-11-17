# data/metadata/

Anonymized index files. Each row is one participant or one scan; no imaging
data is stored here.

| File                             | Rows per | Used by |
|----------------------------------|----------|---------|
| scanners.csv                     | scanner  | Tables X, XI; scanner-disjoint retrain |
| datasets_summary.csv             | corpus   | Table II |
| severity_scores_summary.csv      | site     | Table XII; severity subset |
| abide_participants.csv           | subject  | ABIDE splits, demographics |
| abide2_participants.csv          | subject  | Unseen-site evaluation |
| rest_meta_mdd_participants.csv   | subject  | REST-meta-MDD splits, demographics |
| srpbs_md_participants.csv        | subject  | Cross-dataset test |
| srpbs_ts_sessions.csv            | session  | V3 pair construction |
| hcp_participants.csv             | subject  | V2 pair construction, fingerprint |

The participant-level csvs are rebuilt by `preprocessing/build_metadata.py`
from the raw releases. They are not redistributed here for datasets whose DUA
forbids it; check the dataset's terms before committing them.
