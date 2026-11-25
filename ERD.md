# Sentinel Health — Entity Relationship Design

## 1. Purpose
This ERD is a logical future schema for dataset versions, pipeline executions, experiments, models, metrics, predictions, explanations, and monitoring events. The current prototype uses only a local SQLite prediction-event table and a JSON model registry; it does not implement all entities below. It intentionally avoids storing full raw clinical records.

## 2. ER Diagram
```mermaid
erDiagram
  USER ||--o{ EXPERIMENT : creates
  DATASET_VERSION ||--o{ PIPELINE_RUN : used_in
  DATASET_VERSION ||--o{ EXPERIMENT : used_by
  EXPERIMENT ||--o{ MODEL_VERSION : produces
  MODEL_VERSION ||--o{ MODEL_METRIC : has
  MODEL_VERSION ||--o{ PREDICTION : serves
  PREDICTION ||--o{ PREDICTION_EXPLANATION : explained_by
  PIPELINE_RUN ||--o{ MONITORING_EVENT : emits
  MODEL_VERSION ||--o{ MONITORING_EVENT : monitored_by

  USER {
    uuid user_id PK
    string username
    string email
    string role
    datetime created_at
  }

  DATASET_VERSION {
    uuid dataset_version_id PK
    string dataset_name
    string version
    string fingerprint
    int row_count
    int column_count
    string storage_uri
    datetime created_at
  }

  PIPELINE_RUN {
    uuid pipeline_run_id PK
    uuid dataset_version_id FK
    string pipeline_name
    string status
    string stage
    datetime started_at
    datetime completed_at
    float duration_seconds
    string config_hash
    string error_summary
  }

  EXPERIMENT {
    uuid experiment_id PK
    uuid dataset_version_id FK
    uuid user_id FK
    string experiment_name
    string status
    string validation_strategy
    datetime started_at
    datetime completed_at
  }

  MODEL_VERSION {
    uuid model_version_id PK
    uuid experiment_id FK
    string model_name
    string version
    string artifact_uri
    string preprocessing_uri
    string feature_schema_version
    string lifecycle_stage
    datetime created_at
  }

  MODEL_METRIC {
    uuid metric_id PK
    uuid model_version_id FK
    string metric_name
    float metric_value
    string dataset_partition
    datetime recorded_at
  }

  PREDICTION {
    uuid prediction_id PK
    uuid model_version_id FK
    string request_reference
    float probability
    string risk_band
    datetime predicted_at
    float latency_ms
  }

  PREDICTION_EXPLANATION {
    uuid explanation_id PK
    uuid prediction_id FK
    string feature_name
    float contribution
    int rank
  }

  MONITORING_EVENT {
    uuid monitoring_event_id PK
    uuid pipeline_run_id FK
    uuid model_version_id FK
    string event_type
    string severity
    string description
    string metric_name
    float metric_value
    datetime created_at
  }
```

## 3. Entity Definitions
### USER
Optional application identity and role. In a local portfolio version, authentication can be deferred.

### DATASET_VERSION
Represents a dataset snapshot or fingerprint. Track source, version, row/column counts, and storage reference.

### PIPELINE_RUN
One execution of ingestion, validation, preprocessing, training, or monitoring. It stores status, timing, configuration fingerprint, and failure summary.

### EXPERIMENT
A controlled ML experiment tied to a dataset version and validation strategy.

### MODEL_VERSION
A trained model artifact plus its preprocessing artifact and feature-schema version. Keep these linked so inference uses compatible artifacts.

### MODEL_METRIC
Stores metrics for a named partition (train, CV, validation, test). Test metrics should be clearly identified and not used repeatedly for model selection.

### PREDICTION
Stores prediction metadata, model version, probability, risk band if configured, latency, and a non-identifying request reference.

### PREDICTION_EXPLANATION
Stores ranked feature contributions for a prediction. Avoid storing raw patient values unless required and approved.

### MONITORING_EVENT
Stores quality, drift, pipeline, or service events. Metric name/value can be null for event-only alerts.

## 4. Integrity Rules
- Dataset fingerprint and version should be unique.
- Model versions are unique within model lineage.
- Model metric names and partition names are constrained.
- Predictions must reference an existing model version.
- Explanations must reference a prediction.
- Pipeline runs must reference the dataset version they processed.
- Model and preprocessing artifacts must share a feature-schema version.
- Do not store direct identifiers in prediction records.
- Retention and deletion policy should be documented.

## 5. Suggested SQLAlchemy Model Modules
- `DatasetVersion`
- `PipelineRun`
- `Experiment`
- `ModelVersion`
- `ModelMetric`
- `Prediction`
- `PredictionExplanation`
- `MonitoringEvent`
- Optional `User`
