# ML Engineer — Responsibilities, Learning Path & Design Patterns

---

## What is an ML Engineer? (vs Data Scientist vs Data Engineer)

| Role | Primary Job | Owns |
|------|-------------|------|
| **Data Scientist** | Find insights, build models, experiment | Notebooks, model accuracy |
| **Data Engineer** | Build data pipelines, warehouses | ETL, Spark, Airflow, schemas |
| **ML Engineer** | Take DS models to production and keep them alive | Training pipelines, serving, monitoring, drift |
| **MLOps Engineer** | Infrastructure for ML at scale | Feature stores, ML platforms, CI/CD for ML |

**This project makes you an ML Engineer.** You own all 4 flows: training, serving, observability, deployment.

---

## Responsibilities by Seniority

### Junior ML Engineer (0–2 years) — WHERE YOU ARE NOW

**You own:**
- Writing and debugging training scripts
- Data cleaning and feature engineering
- Basic model evaluation (R², MAE, confusion matrix)
- Running experiments and logging results (MLflow)
- Writing tests for your code
- Using Docker locally

**You should learn:**
- Python (pandas, numpy, sklearn, XGBoost)
- SQL (joins, aggregations, window functions)
- Git (commit, branch, PR, rebase, cherry-pick)
- Docker basics (build, run, compose)
- FastAPI or Flask for serving a model
- MLflow for experiment tracking
- Basic statistics (mean, std, distributions, correlation)
- How to read and write clean code

**You do NOT need yet:**
- Kubernetes (use it, don't build it)
- Distributed training (Ray, Horovod)
- Building a feature store
- System design interviews at senior level
- Kafka, Flink, Spark

---

### Mid-Level ML Engineer (2–4 years)

**You own:**
- Full training pipeline (data → model → export → register)
- Model serving API with caching, batching, monitoring
- Drift detection and automated retraining triggers
- CI/CD for ML (GitHub Actions, Docker build, deploy)
- A/B testing and shadow deployments
- Writing and owning production incidents

**You should learn:**
- Kubernetes basics (pods, deployments, HPA, health checks)
- Helm for deployment
- Prometheus + Grafana for monitoring
- Redis (cache, queue, DLQ)
- PostgreSQL (schema design, indexing, query optimization)
- ONNX / model optimization
- SHAP / model explainability
- Feature engineering at scale
- Data validation (Pandera, Great Expectations)
- System design basics (caching, load balancing, CAP theorem)
- Cloud fundamentals (AWS or GCP — EKS, S3, ECR, Secrets Manager)

**You do NOT need yet:**
- Building your own ML platform
- Triton inference server
- Kubeflow pipelines
- Statistical rigor for A/B testing (p-values, sample size calc)
- Multi-GPU distributed training

---

### Senior ML Engineer (4–7 years)

**You own:**
- End-to-end ML system design from scratch
- Technical decisions: which model, which serving stack, which infra
- Mentoring junior engineers
- Cross-team API contracts
- On-call for ML production systems
- Designing for scale (100× current load)

**You should learn:**
- Feature store architecture (Feast, Tecton, Hopsworks)
- Distributed training (Ray Train, PyTorch DDP)
- Advanced model serving (Triton, Ray Serve, BentoML)
- Statistical A/B testing (sample size, p-values, multiple comparisons)
- Data lineage (OpenLineage, DataHub)
- ML-specific monitoring (Evidently, Arize, WhyLabs)
- System design interviews at senior level
- Causal inference basics
- ML security (model stealing, adversarial inputs, data poisoning)

**You do NOT need yet:**
- Building Ray from scratch
- PhD-level ML research
- Building Kafka from scratch
- Custom CUDA kernels (unless ML infra team)

---

### Staff / Principal ML Engineer (7+ years)

**You own:**
- ML platform architecture for the whole company
- Technical strategy: what does the ML stack look like in 3 years?
- Make-or-buy decisions (build feature store vs buy Tecton)
- Standards across multiple teams
- Hiring bar for ML engineers

**You should learn:**
- How to lead without authority
- Writing technical RFCs and design docs
- Org-level tradeoffs (cost vs velocity vs reliability)
- Advanced distributed systems
- LLM systems (RAG, fine-tuning, evals, guardrails) — increasingly required

---

## Real-Time ML Engineering Design Patterns

These are patterns that repeat across EVERY production ML system.

### Pattern 1 — Two-Phase Prediction
**Problem:** Expensive model, millions of requests per second.  
**Solution:** Cheap model filters candidates first, expensive model runs on top N only.  
**Example:** Google Search: fast retrieval model finds 1000 candidates → expensive ranking model ranks top 10.  
**This project:** Redis cache hit = skip ONNX entirely (phase 1 = cache lookup).

---

### Pattern 2 — Feature Store (Online + Offline)
**Problem:** Training uses batch features. Serving needs real-time features. They diverge → train/serve skew.  
**Solution:** One feature store. Offline store (S3/BigQuery) for training. Online store (Redis) for serving. Same feature logic.  
**This project:** neighbourhood_price_rank computed from train data, stored in pkl, served from memory — manual version of this pattern.

---

### Pattern 3 — Shadow Deployment
**Problem:** How do you know if the new model is better before giving it real traffic?  
**Solution:** Run new model in parallel (shadow), compare outputs, promote only if better.  
**This project:** `src/serving/shadow.py` — challenger runs on every request silently.

---

### Pattern 4 — Champion/Challenger
**Problem:** Continuously improving models need a safe promotion path.  
**Solution:** Champion = current production model. Challenger = new candidate. Gate check → shadow → A/B → canary → champion.  
**This project:** Entire rollout pipeline in `shadow.py`, `ab.py`, `canary.py`.

---

### Pattern 5 — Async Batch Inference
**Problem:** User submits 10,000 listings. Synchronous API would time out.  
**Solution:** Accept the job, return a job ID, process in background, client polls for result.  
**This project:** `POST /v1/predict-batch` → Redis queue → background worker → result stored.

---

### Pattern 6 — Model Versioning + Rollback
**Problem:** New model breaks production. How do you go back?  
**Solution:** Every model version tagged and stored. Rollback = point serving to previous version.  
**This project:** MLflow model registry + S3 versioned artifacts. Rollback = change which ONNX file the pod loads.

---

### Pattern 7 — Drift Detection + Auto-Retrain
**Problem:** Model degrades silently because the world changed.  
**Solution:** Monitor input distribution continuously. Detect shift (PSI). Trigger retrain automatically.  
**This project:** `drift.py` computes PSI, nightly GitHub Actions checks, `retrain.py` triggers if PSI > 0.2.

---

### Pattern 8 — Ground Truth Feedback Loop
**Problem:** You know what you predicted, but did it match reality?  
**Solution:** Join predictions with actual outcomes. Compute live MAE. Alert if degrading.  
**This project:** `ground_truth.py` joins stored predictions with real InsideAirbnb prices monthly.

---

### Pattern 9 — Graceful Degradation
**Problem:** ML model is down or too slow. Don't return 500 to users.  
**Solution:** Fallback to simpler model, cached result, or rule-based estimate.  
**This project:** Redis cache hit = serve cached price even if ONNX session crashes.

---

### Pattern 10 — Online Learning
**Problem:** Batch retraining weekly means model is always 7 days behind.  
**Solution:** Model updates incrementally with each new data point in real time.  
**When you need it:** Fraud detection, recommendations, ads click prediction.  
**Not in this project** — Airbnb prices don't change fast enough to need it.

---

## What NOT to Learn (Trap Topics for Beginners)

| Topic | Why Not Yet |
|-------|-------------|
| Building Kubernetes from scratch | You use K8s, you don't build it. Learn to deploy, not to implement |
| PhD-level deep learning theory | Only matters if you're doing ML research, not ML engineering |
| Custom CUDA kernels | Only needed at GPU inference teams (NVIDIA, OpenAI infra) |
| Hadoop / MapReduce | Legacy. Replaced by Spark, Flink, BigQuery. Don't waste time |
| R language | Python won. R is for academic statistics |
| Manual hyperparameter tuning | Optuna/Ray Tune exist. Don't tune by hand |
| Learning every cloud service | Learn AWS deeply first. GCP/Azure follow the same patterns |
| Memorizing algorithms for LC hard | Focus on system design and ML engineering — more valuable |

---

## The Most Important Skill Nobody Teaches

**Debugging in production.**

You will spend more time answering "why is the model predicting wrong?" than building new features. Learn to:
1. Read structured logs (structlog → Datadog)
2. Trace a single request end-to-end (request_id through every component)
3. Compare prediction distributions before/after a deploy
4. Reproduce a production bug locally
5. Write a post-mortem without blaming people

This project gives you all the infrastructure to do this. Use it.
