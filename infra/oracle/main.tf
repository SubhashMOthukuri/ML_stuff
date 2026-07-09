terraform {
  required_providers {
    oci = {
      source  = "oracle/oci"
      version = "~> 6.0"
    }
    local = {
      source  = "hashicorp/local"
      version = "~> 2.4"
    }
  }
  required_version = ">= 1.5"
}

provider "oci" {
  tenancy_ocid     = var.tenancy_ocid
  user_ocid        = var.user_ocid
  fingerprint      = var.fingerprint
  private_key_path = var.private_key_path
  region           = var.region
}

# ── Data sources ──────────────────────────────────────────────────────────────

data "oci_identity_availability_domains" "ads" {
  compartment_id = var.tenancy_ocid
}

# Latest Oracle Linux 8 x86_64 image — standard compute image, works on E3.Flex
data "oci_core_images" "ol8" {
  compartment_id           = var.compartment_id
  operating_system         = "Oracle Linux"
  operating_system_version = "8"
  shape                    = "VM.Standard.E3.Flex"
  sort_by                  = "TIMECREATED"
  sort_order               = "DESC"

  filter {
    name   = "display_name"
    values = ["^Oracle-Linux-8\\.[0-9]+-[0-9]{4}\\.[0-9]{2}\\.[0-9]{2}-[0-9]+$"]
    regex  = true
  }
}

# ── Networking ────────────────────────────────────────────────────────────────

resource "oci_core_vcn" "main" {
  compartment_id = var.compartment_id
  cidr_block     = "10.0.0.0/16"
  display_name   = "${var.app_name}-vcn"
  dns_label      = "nycairbnb"
}

resource "oci_core_internet_gateway" "igw" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.app_name}-igw"
  enabled        = true
}

resource "oci_core_route_table" "public" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.app_name}-rt"

  route_rules {
    destination       = "0.0.0.0/0"
    network_entity_id = oci_core_internet_gateway.igw.id
  }
}

resource "oci_core_security_list" "app" {
  compartment_id = var.compartment_id
  vcn_id         = oci_core_vcn.main.id
  display_name   = "${var.app_name}-sl"

  # Allow all outbound traffic
  egress_security_rules {
    destination = "0.0.0.0/0"
    protocol    = "all"
  }

  # SSH
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 22
      max = 22
    }
  }

  # HTTP
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 80
      max = 80
    }
  }

  # HTTPS
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 443
      max = 443
    }
  }

  # App port (FastAPI / uvicorn)
  ingress_security_rules {
    protocol = "6"
    source   = "0.0.0.0/0"
    tcp_options {
      min = 8001
      max = 8001
    }
  }

  # Redis (internal only — restricted to VCN CIDR)
  ingress_security_rules {
    protocol = "6"
    source   = "10.0.0.0/16"
    tcp_options {
      min = 6379
      max = 6379
    }
  }
}

resource "oci_core_subnet" "public" {
  compartment_id    = var.compartment_id
  vcn_id            = oci_core_vcn.main.id
  cidr_block        = "10.0.0.0/24"
  display_name      = "${var.app_name}-subnet"
  dns_label         = "pub"
  route_table_id    = oci_core_route_table.public.id
  security_list_ids = [oci_core_security_list.app.id]
}

# ── Compute instance ──────────────────────────────────────────────────────────

resource "oci_core_instance" "app" {
  compartment_id      = var.compartment_id
  availability_domain = data.oci_identity_availability_domains.ads.availability_domains[0].name
  display_name        = var.app_name
  shape               = "VM.Standard.E3.Flex"

  shape_config {
    ocpus         = var.ocpus
    memory_in_gbs = var.memory_gb
  }

  source_details {
    source_type = "image"
    source_id   = data.oci_core_images.ol8.images[0].id
  }

  create_vnic_details {
    subnet_id        = oci_core_subnet.public.id
    assign_public_ip = true
    display_name     = "${var.app_name}-vnic"
  }

  metadata = {
    ssh_authorized_keys = file(pathexpand(var.ssh_public_key_path))
    user_data           = base64encode(local.cloud_init)
  }
}

# ── Cloud-init: install Docker + Docker Compose on first boot ─────────────────

locals {
  cloud_init = <<-EOF
    #!/bin/bash
    set -e

    # Install Docker
    dnf install -y dnf-utils
    dnf config-manager --add-repo https://download.docker.com/linux/centos/docker-ce.repo
    dnf install -y docker-ce docker-ce-cli containerd.io docker-compose-plugin
    systemctl enable --now docker

    # Allow opc user to run docker without sudo
    usermod -aG docker opc

    # Create app directory
    mkdir -p /opt/nyc-airbnb
    chown opc:opc /opt/nyc-airbnb

    echo "VM ready. SSH in and run: cd /opt/nyc-airbnb && docker compose up -d"
  EOF
}

