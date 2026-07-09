#!/usr/bin/env bash
# Upgrades the EKS cluster one minor version at a time.
# Called by .github/workflows/eks-upgrade.yml for each version step.
#
# Usage: eks_step_upgrade.sh <target_version> <dry_run>
# Example: eks_step_upgrade.sh 1.32 true
#
# Env vars (set by the calling workflow):
#   CLUSTER     — EKS cluster name   (default: nyc-airbnb)
#   REGION      — AWS region          (default: us-east-2)
#   NODE_GROUP  — managed node group  (default: nyc-airbnb-nodes)

set -euo pipefail

TARGET=${1:?'target version required'}
DRY_RUN=${2:-true}
CLUSTER=${CLUSTER:-nyc-airbnb}
REGION=${REGION:-us-east-2}
NODE_GROUP=${NODE_GROUP:-nyc-airbnb-nodes}
ADDON_WAIT_RETRIES=20
CLUSTER_WAIT_RETRIES=60   # 60 × 30 s = 30 min
NG_WAIT_RETRIES=60

CURRENT=$(aws eks describe-cluster --name "$CLUSTER" --region "$REGION" \
  --query 'cluster.version' --output text)

if [[ "$CURRENT" == "$TARGET" ]]; then
  echo "Cluster already at $TARGET — skipping step"
  exit 0
fi

echo "==> Step: EKS $CURRENT → $TARGET  (dry_run=$DRY_RUN)"

# ── Dry run ───────────────────────────────────────────────────────────────────

if [[ "$DRY_RUN" == "true" ]]; then
  echo "[DRY RUN] would upgrade control plane $CURRENT → $TARGET"
  echo "[DRY RUN] would upgrade add-ons (vpc-cni, kube-proxy, coredns) to $TARGET defaults"
  echo "[DRY RUN] would upgrade node group $NODE_GROUP → $TARGET"
  exit 0
fi

# ── 1. Control plane upgrade ──────────────────────────────────────────────────

echo "Upgrading control plane to $TARGET..."
aws eks update-cluster-version \
  --name "$CLUSTER" \
  --kubernetes-version "$TARGET" \
  --region "$REGION" \
  --output text

echo "Waiting for control plane ACTIVE (up to 30 min)..."
for i in $(seq 1 $CLUSTER_WAIT_RETRIES); do
  STATUS=$(aws eks describe-cluster --name "$CLUSTER" --region "$REGION" \
    --query 'cluster.status' --output text)
  printf "  [%02d/%d] status: %s\n" "$i" "$CLUSTER_WAIT_RETRIES" "$STATUS"
  [[ "$STATUS" == "ACTIVE" ]] && break
  if [[ $i -eq $CLUSTER_WAIT_RETRIES ]]; then
    echo "ERROR: Timed out waiting for cluster ACTIVE at $TARGET"
    exit 1
  fi
  sleep 30
done
echo "Control plane is ACTIVE at $TARGET"

# ── 2. Add-on upgrades ────────────────────────────────────────────────────────
# Upgrade after control plane, before node group — this order is required.

for ADDON in vpc-cni kube-proxy coredns; do
  # Skip if not installed
  INSTALLED=$(aws eks describe-addon \
    --cluster-name "$CLUSTER" --addon-name "$ADDON" --region "$REGION" \
    --query 'addon.status' --output text 2>/dev/null || echo "NOT_FOUND")
  if [[ "$INSTALLED" == "NOT_FOUND" ]]; then
    echo "  $ADDON not installed — skipping"
    continue
  fi

  # Get default version for this K8s version
  DEFAULT_VER=$(aws eks describe-addon-versions \
    --kubernetes-version "$TARGET" \
    --addon-name "$ADDON" \
    --region "$REGION" \
    --query 'addons[0].addonVersions[?compatibilities[?defaultVersion==`true`]].addonVersion | [0]' \
    --output text 2>/dev/null || echo "None")

  if [[ -z "$DEFAULT_VER" || "$DEFAULT_VER" == "None" ]]; then
    echo "  $ADDON: could not determine default version for $TARGET — skipping"
    continue
  fi

  CURRENT_VER=$(aws eks describe-addon \
    --cluster-name "$CLUSTER" --addon-name "$ADDON" --region "$REGION" \
    --query 'addon.addonVersion' --output text)

  if [[ "$CURRENT_VER" == "$DEFAULT_VER" ]]; then
    echo "  $ADDON already at $DEFAULT_VER — skipping"
    continue
  fi

  echo "  Upgrading $ADDON: $CURRENT_VER → $DEFAULT_VER"
  aws eks update-addon \
    --cluster-name "$CLUSTER" \
    --addon-name "$ADDON" \
    --addon-version "$DEFAULT_VER" \
    --resolve-conflicts OVERWRITE \
    --region "$REGION" \
    --output text

  for j in $(seq 1 $ADDON_WAIT_RETRIES); do
    ADDON_STATUS=$(aws eks describe-addon \
      --cluster-name "$CLUSTER" --addon-name "$ADDON" --region "$REGION" \
      --query 'addon.status' --output text)
    [[ "$ADDON_STATUS" == "ACTIVE" ]] && break
    if [[ $j -eq $ADDON_WAIT_RETRIES ]]; then
      echo "  WARNING: $ADDON did not reach ACTIVE — continuing anyway"
      break
    fi
    sleep 15
  done
  echo "  $ADDON is ACTIVE"
done

# ── 3. Managed node group upgrade ────────────────────────────────────────────

echo "Upgrading node group $NODE_GROUP to $TARGET..."
aws eks update-nodegroup-version \
  --cluster-name "$CLUSTER" \
  --nodegroup-name "$NODE_GROUP" \
  --kubernetes-version "$TARGET" \
  --region "$REGION" \
  --output text

echo "Waiting for node group ACTIVE (up to 30 min)..."
for i in $(seq 1 $NG_WAIT_RETRIES); do
  NG_STATUS=$(aws eks describe-nodegroup \
    --cluster-name "$CLUSTER" \
    --nodegroup-name "$NODE_GROUP" \
    --region "$REGION" \
    --query 'nodegroup.status' --output text)
  printf "  [%02d/%d] status: %s\n" "$i" "$NG_WAIT_RETRIES" "$NG_STATUS"
  [[ "$NG_STATUS" == "ACTIVE" ]] && break
  if [[ $i -eq $NG_WAIT_RETRIES ]]; then
    echo "ERROR: Timed out waiting for node group ACTIVE at $TARGET"
    exit 1
  fi
  sleep 30
done
echo "Node group is ACTIVE at $TARGET"
echo "==> Successfully completed step: $CURRENT → $TARGET"
