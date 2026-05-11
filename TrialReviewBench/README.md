---
license: apache-2.0
config_names:
- reviews
- study_search_screen
- data_extraction
configs:
- config_name: reviews
  data_files:
  - split: test
    path: TrialReviewBench-reviews.csv
- config_name: study_search_screen
  data_files:
  - split: test
    path: TrialReviewBench-study-search-screening.jsonl
- config_name: data_extraction
  data_files:
  - split: test
    path: TrialReviewBench-data-extraction/*.csv
---
