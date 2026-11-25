# Sentinel Health CKD Entity Relationships

```mermaid
erDiagram
  DATASET_RUN ||--o{ CANDIDATE_METRIC : evaluates
  DATASET_RUN ||--|| TRAINING_REPORT : produces
  TRAINING_REPORT ||--|| MODEL_VERSION : selects
  MODEL_VERSION ||--o{ SCORE_EVENT : serves
  MODEL_VERSION ||--o{ ARTIFACT : emits
```

The application stores model/run metadata and aggregate score telemetry. It does not store patient-level prediction inputs. The included source dataset is read locally and model artifacts remain under `artifacts/ckd/`.
