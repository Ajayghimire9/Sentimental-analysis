# Cloud infrastructure extension

This directory is intentionally a cloud-ready boundary rather than a fake production environment. Credentials, account IDs, regions, registries, networking, and managed Kubernetes configuration belong to the deployment environment.

Recommended target architecture:

```text
GitHub Actions
   ↓
Container registry
   ↓
Managed Kubernetes
   ├── FastAPI sentiment service
   ├── Prometheus/Grafana
   └── MLflow
          ↓
     Object storage
```

A production implementation should provision these resources with environment-specific variables and remote Terraform state. No credentials are committed to this repository.
