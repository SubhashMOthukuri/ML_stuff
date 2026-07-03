# ─────────────────────────────────────────────────────────────────────────────
# outputs.tf — What Terraform prints when it finishes building
#
# Like the address label on your finished laminate station — tells you
# where it is and how to reach it.
# ─────────────────────────────────────────────────────────────────────────────

output "cluster_id" {
  description = "OKE cluster OCID — used by Helm to deploy the app"
  value       = oci_containerengine_cluster.main.id
}

output "cluster_name" {
  description = "Kubernetes cluster name"
  value       = oci_containerengine_cluster.main.name
}

output "cluster_endpoint" {
  description = "K8s API server URL — what kubectl connects to"
  value       = oci_containerengine_cluster.main.endpoints[0].public_endpoint
}

output "kubeconfig_path" {
  description = "Path to the kubeconfig file — run: export KUBECONFIG=$(terraform output -raw kubeconfig_path)"
  value       = local_file.kubeconfig.filename
}

output "node_pool_id" {
  description = "ARM node pool OCID"
  value       = oci_containerengine_node_pool.arm_workers.id
}

output "vcn_id" {
  description = "VCN OCID — needed if you add more subnets later"
  value       = oci_core_vcn.main.id
}

output "lb_subnet_id" {
  description = "Load balancer subnet — Oracle creates the LB here when you deploy an Ingress"
  value       = oci_core_subnet.lb.id
}

output "next_steps" {
  description = "What to do after terraform apply finishes"
  value       = <<-EOT

    ── Cluster is ready ──────────────────────────────────────────
    1. Point kubectl at your new cluster:
       export KUBECONFIG=${local_file.kubeconfig.filename}

    2. Verify nodes are Ready (takes ~3 min after apply):
       kubectl get nodes

    3. Deploy the app with Helm:
       cd ../helm && helm install nyc-airbnb ./nyc-airbnb

    4. When done for the day — destroy everything (pay $0):
       terraform destroy
    ──────────────────────────────────────────────────────────────
  EOT
}
