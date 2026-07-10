variable "tenancy_ocid" {
  description = "Your Oracle account root ID"
  type        = string
}

variable "user_ocid" {
  description = "Your Oracle user ID"
  type        = string
}

variable "fingerprint" {
  description = "Fingerprint of the API signing key"
  type        = string
}

variable "private_key_path" {
  description = "Path to OCI private key file"
  type        = string
  default     = "~/.oci/oci_api_key.pem"
}

variable "region" {
  description = "Oracle region"
  type        = string
  default     = "us-chicago-1"
}

variable "compartment_id" {
  description = "Oracle compartment to deploy into"
  type        = string
}

variable "app_name" {
  description = "Name prefix for all resources"
  type        = string
  default     = "nyc-airbnb"
}

# ── VM sizing: E3.Flex ~$0.025/OCPU/hr + $0.0015/GB/hr ──────────────────────
# 2 OCPU + 16 GB ≈ $1.20/day

variable "ocpus" {
  description = "OCPUs for the VM"
  type        = number
  default     = 2
}

variable "memory_gb" {
  description = "GB RAM for the VM"
  type        = number
  default     = 16
}

variable "ssh_public_key_path" {
  description = "Path to SSH public key for VM access"
  type        = string
  default     = "~/.ssh/id_rsa.pub"
}

variable "ssh_allowed_cidr" {
  description = "CIDR allowed to SSH into the instance. Restrict to your static IP (e.g. '203.0.113.5/32'). Default allows all — change before production use."
  type        = string
  default     = "0.0.0.0/0"
}

# ── Vault secrets (kept for app env vars) ─────────────────────────────────────

variable "vault_id" {
  description = "OCID of the Oracle Vault"
  type        = string
}

variable "vault_key_id" {
  description = "OCID of the vault master key"
  type        = string
}

variable "secret_redis_password_id" {
  type = string
}

variable "secret_valid_api_keys_id" {
  type = string
}

variable "secret_slack_webhook_id" {
  type = string
}

variable "secret_dd_api_key_id" {
  type = string
}

variable "secret_ground_truth_ingest_id" {
  type = string
}
