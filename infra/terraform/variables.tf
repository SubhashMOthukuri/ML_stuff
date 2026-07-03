# ─────────────────────────────────────────────────────────────────────────────
# variables.tf — Every value the blueprint can accept
#
# Think of these as the knobs on a laminate machine — you can adjust them
# without rewriting the whole machine design.
# Actual values live in terraform.tfvars (gitignored).
# ─────────────────────────────────────────────────────────────────────────────

# ── Who you are ───────────────────────────────────────────────────────────────

variable "tenancy_ocid" {
  description = "Your Oracle account root ID"
  type        = string
}

variable "user_ocid" {
  description = "Your Oracle user ID"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the API signing key you uploaded"
  type        = string
}

variable "private_key_path" {
  description = "Path to your local OCI private key file"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "region" {
  description = "Oracle region to deploy everything in"
  type        = string
  default     = "us-chicago-1"
}

variable "compartment_id" {
  description = "Oracle compartment (folder) to put resources in"
  type        = string
}

# ── What to build ─────────────────────────────────────────────────────────────

variable "cluster_name" {
  description = "Name of the Kubernetes cluster"
  type        = string
  default     = "nyc-airbnb"
}

variable "kubernetes_version" {
  description = "Kubernetes version for the cluster and nodes"
  type        = string
  default     = "v1.30.1"
}

# ── Free tier ARM nodes ────────────────────────────────────────────────────────
# VM.Standard.A1.Flex = Oracle free ARM shape (Ampere A1)
# Free allowance: 4 OCPUs + 24 GB RAM total
# 2 nodes × 2 OCPUs × 12 GB = stays within free tier

variable "node_count" {
  description = "Number of ARM worker nodes"
  type        = number
  default     = 2
}

variable "node_ocpus" {
  description = "OCPUs per node (2 × 2 = 4 total, free tier limit)"
  type        = number
  default     = 2
}

variable "node_memory_gb" {
  description = "GB RAM per node (2 × 12 = 24 GB total, free tier limit)"
  type        = number
  default     = 12
}

# ── Vault (created manually, referenced here) ─────────────────────────────────

variable "vault_id" {
  description = "OCID of the Oracle Vault we created"
  type        = string
}

variable "vault_key_id" {
  description = "OCID of the master encryption key inside the vault"
  type        = string
}

variable "secret_redis_password_id" {
  description = "OCID of the REDIS_PASSWORD secret in Oracle Vault"
  type        = string
}

variable "secret_valid_api_keys_id" {
  description = "OCID of the VALID_API_KEYS secret in Oracle Vault"
  type        = string
}

variable "secret_slack_webhook_id" {
  description = "OCID of the SLACK_WEBHOOK_URL secret in Oracle Vault"
  type        = string
}

variable "secret_dd_api_key_id" {
  description = "OCID of the DD_API_KEY secret in Oracle Vault"
  type        = string
}

variable "secret_ground_truth_ingest_id" {
  description = "OCID of the GROUND_TRUTH_INGEST_TOKEN secret in Oracle Vault"
  type        = string
}
