# Project Audit

## Overview
This repository is a Flask-based keyword spotting application that loads audio, extracts MFCC features, trains a CNN-LSTM model, tracks experiments with MLflow, and exposes inference through a web UI. The project is designed to run as a Dockerized service and deploy to AWS with GitHub Actions CI/CD.

## Current repository state
The workspace contains the key application structure expected for this project:

- app.py
- Dockerfile
- README.md
- pyproject.toml
- requirements.txt
- config_dir/
- dataset/
- src/
- static/
- templates/
- tests/
- artifacts/

## What exists

### Python application files
- app.py: Flask web entrypoint
- src/data.py: dataset loading / MFCC processing
- src/inference.py: model loading and prediction
- src/main.py: training pipeline entrypoint
- src/model.py: CNN-LSTM architecture
- src/train.py: training loop
- src/experiment_tracking.py: MLflow experiment tracking
- src/exception_handler.py: custom exception classes

### Config
- config_dir/config.yaml
- config_dir/configType.py

### Frontend assets
- templates/page.html
- static/page.css
- static/bg.jpg

### Test suite
- tests/test_kws_spotter.py

### Data
- dataset/test/ (sample WAV files)
- dataset/train/labels.txt

### MLflow / artifacts
- artifacts/ (model directory created during local validation)

### Docker config
- Dockerfile

### Python packaging
- pyproject.toml
- requirements.txt
- poetry.lock

## What is missing or incomplete
- No real GitHub Actions workflows existed initially in the repository.
- No working AWS deployment automation was present.
- No full Google Speech Commands dataset was included; only a minimal sample dataset exists in the current workspace.
- No pre-trained production model artifact was present before local validation.
- The repository was using outdated Hydra interpolation patterns that were incompatible with plain Python config loading.
- Dockerfile was pinned to Python 3.8 and Poetry 1.1.x, which is outdated for the current environment.

## Broken or incompatible items found

### Python / dependency compatibility
- Original project assumptions were designed around older Python / TensorFlow / MLflow / librosa stack.
- The project used Hydra interpolation such as `${hydra:runtime.cwd}` without Hydra runtime context.
- librosa usage needed compatibility adjustments for current APIs.
- The original MLflow model loading path assumed a tracked MLflow artifact URI; the local saved model is a standard Keras model directory and must be loaded directly.

### Configuration problems
- Config file used Hydra interpolation that failed when OmegaConf was loaded outside a Hydra application context.
- Relative path assumptions were incompatible on Windows without explicit project-root-aware logic.

### Dataset problems
- Dataset is not a full Google Speech Commands dataset in this workspace.
- Only a small sample of WAV files and labels are present.
- The training pipeline expects more complete data than the repo currently contains.

### Model problems
- Model artifacts were not populated in the repository state before local validation.
- The app and inference path required a model directory that exists before requests can be served.

### Docker problems
- Dockerfile references Python 3.8 and legacy Poetry steps.
- The project should be containerized using a stable Python 3.11 base and the current dependency manifest (`requirements.txt`) rather than a stale Poetry-only install path.

### CI/CD problems
- GitHub Actions workflows were missing.
- AWS auth could not be configured without OIDC and repository secrets.
- No deployment pipeline for Amazon ECR or EC2 existed.

## Environment and runtime findings
The local project was validated under:
- Python 3.11
- TensorFlow 2.15.1
- librosa 0.10.2.post1
- Flask 3.0.3
- MLflow 2.15.1
- OmegaConf 2.3.0
- pytest 8.3.3

This combination was selected because it is stable for the project’s existing ML stack and works on modern Windows environments.

## Validation performed
- OmegaConf config load check.
- Basic MFCC extraction validation.
- pytest validation: 3 tests passed.
- Local Flask app boot: HTTP 200 confirmed.
- Local Keras model inference path validated using a generated sample model artifact.

## Recommended operational path
1. Keep the current Python 3.11 virtual environment for local dev.
2. Use requirements.txt as the runtime manifest for Docker and CI.
3. Keep the project architecture intact; do not replace the ML model concept.
4. Use GitHub Actions with OIDC for AWS access.
5. Push Docker image to Amazon ECR.
6. Deploy the container to EC2 using SSH or SSM.
7. Use repo secrets for AWS role, ECR repository, EC2 host, and SSH key.

## AWS deployment requirements
To complete AWS deployment, the user must provide:
- AWS account ID
- AWS region
- ECR repository name
- EC2 instance public DNS or IP
- EC2 SSH user (typically `ec2-user` or `ubuntu`)
- IAM role ARN for GitHub OIDC
- GitHub repository secrets for AWS role and SSH key

Never hardcode AWS access keys or secrets in source code.
Use IAM roles and GitHub OIDC where possible.

## Final assessment
The project is workable in a modern local Windows environment after compatibility fixes, and the application can be validated locally. The remaining deployment work is infrastructure-specific and requires AWS credentials / EC2 access details from the user, not code changes alone.
