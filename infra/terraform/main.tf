# ─────────────────────────────────────────────────────────────────────────────
# main.tf — The actual blueprint
#
# Reading order:
#   1. provider  → tells Terraform "talk to Oracle"
#   2. data      → looks up existing Oracle info (available zones, OS images)
#   3. network   → builds the private office network (VCN + subnets)
#   4. cluster   → creates the Kubernetes control plane
#   5. nodes     → adds ARM worker machines to the cluster
#   6. kubeconfig→ saves the "key" kubectl needs to talk to the cluster
# ─────────────────────────────────────────────────────────────────────────────

terraform {
  required_providers {
    # oci = Oracle Cloud Infrastructure provider
    # Like an adapter plug — Terraform speaks generic, this translates to Oracle's API
    oci = {
      source  = "oracle/oci"
      version = "~> 6.0"
    }
    # local = writes files to your machine (used for kubeconfig)
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
  required_version = ">= 1.5"
}

# ── Provider: tell Terraform who you are and which cloud to use ───────────────
provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# ─────────────────────────────────────────────────────────────────────────────
# DATA SOURCES — look up existing Oracle information
# "data" = read-only, nothing is created
# ─────────────────────────────────────────────────────────────────────────────

# Which physical data centre zones are available in our region?
# us-chicago-1 has 3 availability domains (AD-1, AD-2, AD-3)
# We put worker nodes in AD-1 — if AD-1 has no free ARM capacity, change to [1] or [2]
data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Find the latest Oracle Linux 8 image that works on ARM (A1) shape
# Why dynamic lookup? Oracle releases new patched images monthly.
# Hardcoding an image ID means you get security patches never — bad.
data "oci_core_images" "arm_linux" {
  compartment_id           = var.compartment_id
  operating_system         = "Oracle Linux"
  operating_system_version = "8"
  shape                    = "VM.Standard.A1.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"

  filter {
    name   = "display_name"
    values = ["^.*aarch64.*$"]   # aarch64 = ARM 64-bit architecture
    regex  = true
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# NETWORKING — the private office network
#
# Analogy: think of VCN as your office building.
#   - The building has a private floor (workers, internal traffic)
#   - The building has a public reception (load balancer, public traffic)
#   - The building has a control room (K8s API endpoint)
#   - Internet Gateway = the front door (public traffic in/out)
#   - NAT Gateway = the back door (workers can call the internet, internet can't call them)
# ─────────────────────────────────────────────────────────────────────────────

# The building itself — 10.0.0.0/16 = 65,536 private IP addresses to work with
resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_id
  cidr_block     = "10.0.0.0/16"
  display_name   = "${var.cluster_name}-vcn"
  dns_label      = "nycairbnb"
}

# Front door — lets public subnets send/receive internet traffic
resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-igw"
  enabled        = true
}

# Back door — worker nodes pull Docker images from the internet,
# but nothing on the internet can directly reach the workers
resource "oci_core_nat_gateway" "nat" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-nat"
}

# Oracle services gateway — lets workers reach Oracle services (object storage etc.)
# without going out to the public internet
resource "oci_core_service_gateway" "svc" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-svc-gw"

  services {
    service_id = data.oci_core_services.all.services[0].id
  }
}

data "oci_core_services" "all" {
  filter {
    name   = "name"
    values = ["All .* Services In Oracle Services Network"]
    regex  = true
  }
}

# Route table for PUBLIC subnets (endpoint + load balancer)
# Rule: anything going to the internet → use the front door (internet gateway)
resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-public-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

# Route table for PRIVATE subnets (worker nodes)
# Rule 1: Oracle services → use service gateway (no internet)
# Rule 2: everything else → use NAT gateway (workers can reach internet, not vice versa)
resource "oci_core_route_table" "private" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-private-rt"

  route_rules {
    destination_type  = "SERVICE_CIDR_BLOCK"
    destination       = data.oci_core_services.all.services[0].cidr_block
    network_entity_id = oci_core_service_gateway.svc.id
  }

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_nat_gateway.nat.id
  }
}

# ── Security lists (firewall rules) ──────────────────────────────────────────

# Firewall for the K8s API endpoint subnet
# Allow: kubectl on port 6443 from anywhere, all traffic within VCN
resource "oci_core_security_list" "endpoint" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-endpoint-sl"

  # Allow kubectl to reach K8s API server from your laptop
  ingress_security_rules {
    protocol = "6"   # TCP
    source   = "0.0.0.0/0"
    tcp_options {
      min = 6443
      max = 6443
    }
  }

  # Allow internal VCN traffic (node ↔ control plane communication)
  ingress_security_rules {
    protocol = "6"
    source   = "10.0.0.0/16"
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# Firewall for the load balancer subnet
# Allow: HTTP (80) and HTTPS (443) from the internet
resource "oci_core_security_list" "lb" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-lb-sl"

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# Firewall for worker nodes
# Allow: all traffic within VCN + NodePort range for K8s services
resource "oci_core_security_list" "workers" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.cluster_name}-workers-sl"

  # All internal VCN traffic (pods talking to each other, control plane → nodes)
  ingress_security_rules {
    protocol = "all"
    source   = "10.0.0.0/16"
  }

  # NodePort range — K8s exposes services here when type=NodePort
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 30000
      max = 32767
    }
  }

  egress_security_rules {
    protocol    = "all"
    destination = "0.0.0.0/0"
  }
}

# ── Subnets — floors of the building ─────────────────────────────────────────

# K8s API endpoint — small subnet, only needs IPs for the control plane
resource "oci_core_subnet" "endpoint" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  cidr_block                 = "10.0.0.0/28"   # 14 usable IPs — plenty for control plane
  display_name               = "${var.cluster_name}-endpoint"
  dns_label                  = "endpoint"
  prohibit_public_ip_on_vnic = false            # public IP so kubectl can reach it
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.endpoint.id]
}

# Load balancer — where public traffic enters the cluster
resource "oci_core_subnet" "lb" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  cidr_block                 = "10.0.1.0/24"   # 254 usable IPs
  display_name               = "${var.cluster_name}-lb"
  dns_label                  = "lb"
  prohibit_public_ip_on_vnic = false            # public IP for the load balancer
  route_table_id             = oci_core_route_table.public.id
  security_list_ids          = [oci_core_security_list.lb.id]
}

# Worker nodes — private, no direct internet exposure
resource "oci_core_subnet" "workers" {
  compartment_id             = var.compartment_id
  vcn_id                     = oci_core_vcn.main.id
  cidr_block                 = "10.0.2.0/24"   # 254 usable IPs
  display_name               = "${var.cluster_name}-workers"
  dns_label                  = "workers"
  prohibit_public_ip_on_vnic = true             # NO public IP — workers are private
  route_table_id             = oci_core_route_table.private.id
  security_list_ids          = [oci_core_security_list.workers.id]
}

# ─────────────────────────────────────────────────────────────────────────────
# KUBERNETES CLUSTER — the control plane
#
# This is the "brain" of Kubernetes — it decides which pod runs on which node,
# watches for failures, and exposes the API kubectl talks to.
# Oracle manages this for you (OKE = Oracle Kubernetes Engine).
# You pay $0 for the control plane on Oracle free tier.
# ─────────────────────────────────────────────────────────────────────────────

resource "oci_containerengine_cluster" "main" {
  compartment_id     = var.compartment_id
  name               = var.cluster_name
  kubernetes_version = var.kubernetes_version
  vcn_id             = oci_core_vcn.main.id

  # Put the K8s API server in the endpoint subnet with a public IP
  # so you can run kubectl from your laptop
  endpoint_config {
    is_public_ip_enabled = true
    subnet_id            = oci_core_subnet.endpoint.id
  }

  options {
    # Tell K8s where to create load balancers when you deploy a Service
    service_lb_subnet_ids = [oci_core_subnet.lb.id]

    add_ons {
      is_kubernetes_dashboard_enabled = false
      is_tiller_enabled               = false
    }

    # Internal K8s DNS — pods can find each other by name (e.g. "redis.default.svc.cluster.local")
    kubernetes_network_config {
      pods_cidr     = "10.244.0.0/16"
      services_cidr = "10.96.0.0/16"
    }
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# NODE POOL — the worker machines
#
# These are the actual ARM servers that run your pods.
# Shape VM.Standard.A1.Flex = Oracle's free Ampere A1 ARM processors.
# Free tier: up to 4 OCPUs + 24 GB RAM total across all A1 instances.
# ─────────────────────────────────────────────────────────────────────────────

resource "oci_containerengine_node_pool" "arm_workers" {
  cluster_id         = oci_containerengine_cluster.main.id
  compartment_id     = var.compartment_id
  name               = "arm-workers"
  kubernetes_version = var.kubernetes_version

  # ARM shape — free tier
  node_shape = "VM.Standard.A1.Flex"

  node_shape_config {
    ocpus         = var.node_ocpus
    memory_in_gbs = var.node_memory_gb
  }

  # Latest Oracle Linux 8 ARM image (looked up dynamically above)
  node_source_details {
    source_type = "IMAGE"
    image_id    = data.oci_core_images.arm_linux.images[0].id
  }

  node_config_details {
    size = var.node_count

    # Spread nodes across all 3 ADs so Terraform picks whichever has A1 capacity.
    # ARM A1 free-tier inventory is limited per AD — having all 3 means OKE can
    # place the node in whichever AD currently has capacity.
    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
      subnet_id           = oci_core_subnet.workers.id
    }

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[1].name
      subnet_id           = oci_core_subnet.workers.id
    }

    placement_configs {
      availability_domain = data.oci_identity_availability_domains.ads.availability_domains[2].name
      subnet_id           = oci_core_subnet.workers.id
    }
  }

  # Kubernetes labels on each node — useful for targeting specific nodes
  # e.g. in a pod spec: nodeSelector: { role: worker }
  initial_node_labels {
    key   = "role"
    value = "worker"
  }

  initial_node_labels {
    key   = "arch"
    value = "arm64"
  }
}

# ─────────────────────────────────────────────────────────────────────────────
# KUBECONFIG — the "key" kubectl uses to talk to your cluster
#
# After the cluster is built, we fetch the kubeconfig and save it locally.
# Then: export KUBECONFIG=infra/terraform/kubeconfig && kubectl get nodes
# ─────────────────────────────────────────────────────────────────────────────

data "oci_containerengine_cluster_kube_config" "kube" {
  cluster_id = oci_containerengine_cluster.main.id
}

resource "local_file" "kubeconfig" {
  content         = data.oci_containerengine_cluster_kube_config.kube.content
  filename        = "${path.module}/kubeconfig"
  file_permission = "0600"   # owner-only read — same as SSH keys
}
