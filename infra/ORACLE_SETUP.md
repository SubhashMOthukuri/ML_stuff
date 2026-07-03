# Oracle Cloud Setup — What We Did & Why

## The Big Picture

We are moving our NYC Airbnb Price Prediction API from a local Docker Compose setup
to a real cloud infrastructure on Oracle Cloud. We chose Oracle because it has a
genuinely free tier that never expires — including free Kubernetes, free ARM servers,
and a free load balancer.

This document records every step we took and the reason behind each one.

---

## Step 1 — Installed the Oracle CLI (`oci`)

**Command:**
```bash
brew install oci-cli
```

**Why:**
The OCI CLI is a remote control for your Oracle account from the terminal.
Instead of clicking through the Oracle website every time, you type one command
and it does the work. Terraform and Kubernetes also use it under the hood.

---

## Step 2 — Connected the CLI to your Oracle account (`oci setup config`)

**Command:**
```bash
oci setup config
```

**What it asked and what we answered:**

| Question | Answer | Why |
|----------|--------|-----|
| Config location | (Enter — default) | CLI always looks here |
| User OCID | From Oracle dashboard → My Profile | Tells Oracle which user you are |
| Tenancy OCID | From Oracle dashboard → Tenancy | Tells Oracle which account/company |
| Region | `us-chicago-1` | Your account's home region |
| Generate new key? | Y (Enter) | Creates an RSA key pair for signing requests |
| Key directory | (Enter — default `~/.oci/`) | Standard location |
| Key name | (Enter — default `oci_api_key`) | Standard name |
| Passphrase | `N/A` | No passphrase — CLI needs to read key automatically without human input |

**What it created:**
- `~/.oci/oci_api_key.pem` — your private key (never share this)
- `~/.oci/oci_api_key_public.pem` — your public key (safe to share)
- `~/.oci/config` — config file the CLI reads every time

**Why RSA keys instead of a password?**
Oracle doesn't use username + password for API calls. Instead, every request
you make is *signed* with your private key. Oracle verifies the signature using
your public key. This means even if someone intercepts the request, they can't
fake or modify it without your private key.

---

## Step 3 — Uploaded the public key to Oracle Console

**Why:**
Oracle needs your public key to verify your signed requests. Without it,
every CLI command would be rejected with "unauthorized".

**Steps in Oracle Console:**
1. Avatar (top right) → My Profile
2. API Keys → Add API Key
3. Paste Public Key → pasted contents of `~/.oci/oci_api_key_public.pem`
4. Oracle showed fingerprint: `83:bb:52:d6:60:73:74:5e:21:16:20:bb:8c:4f:3b:91`
   — this matched what the CLI generated, confirming the right key was uploaded.

---

## Step 4 — Verified the connection

**Command:**
```bash
oci iam region list --output table
```

**Why:**
If this prints a table of regions, the CLI can talk to Oracle.
If it fails, something in the config or key upload went wrong.
We got a full table of 44 regions — connection confirmed ✅

---

## Step 5 — Created Oracle Vault

**What is Oracle Vault?**
A Vault is a secure safe for secrets (passwords, API keys, tokens).
Without a Vault, secrets live in `.env` files or environment variables —
plain text that anyone with file access can read.
With a Vault, secrets are encrypted using Oracle's hardware chips.
Your app asks the Vault at runtime: "give me the Redis password" —
the password never sits in a file.

```
Without Vault:   REDIS_PASSWORD=mypassword  ← anyone can read this file
With Vault:      app → asks Oracle Vault → Oracle checks permissions → returns secret
```

**Command:**
```bash
oci kms management vault create \
  --compartment-id ocid1.tenancy.oc1..aaaaaaaai3zdlbkwcou4zu27p5eegciy2n3olfcl7iilimsl77uyxch4bktq \
  --display-name nyc-airbnb-vault \
  --vault-type DEFAULT \
  --region us-chicago-1
```

**Result:**
- Vault ID: `ocid1.vault.oc1.us-chicago-1.ijvenqkpaacq2.abxxeljtva3627tetbkqf2gavwnelxfmuofetyye6udixdsu3f6el6jgg2uq`
- Management endpoint: `https://ijvenqkpaacq2-management.kms.us-chicago-1.oci.oraclecloud.com`

---

## Step 6 — Created a Master Encryption Key

**What is a Master Key?**
The master key is the combination lock on the vault safe.
Every secret you store is encrypted using this key.
Oracle stores the key inside a hardware security module (HSM) —
a physical chip that is designed to never expose the raw key.
You never see the key itself, Oracle's hardware manages it.

**Command:**
```bash
oci kms management key create \
  --compartment-id ocid1.tenancy.oc1..aaaaaaaai3zdlbkwcou4zu27p5eegciy2n3olfcl7iilimsl77uyxch4bktq \
  --display-name nyc-airbnb-master-key \
  --key-shape '{"algorithm":"AES","length":32}' \
  --endpoint https://ijvenqkpaacq2-management.kms.us-chicago-1.oci.oraclecloud.com \
  --region us-chicago-1
```

**Result:**
- Key ID: `ocid1.key.oc1.us-chicago-1.ijvenqkpaacq2.abxxeljsy7rpy6pcnlgdginrogj4hlqfkmz752hwktoqqh3sa3y3bvqppxjq`
- Algorithm: AES-256 (industry standard for symmetric encryption)

---

## Step 7 — Stored 3 Secrets in the Vault

**What are secrets?**
Secrets are the actual sensitive values locked inside the vault.
Each secret has a name (what your app calls it) and a value (the actual password/key).

| Secret Name | What it protects | OCID |
|-------------|-----------------|------|
| `REDIS_PASSWORD` | Password for Redis cache | `ocid1.vaultsecret.oc1.us-chicago-1.amaaaaaakve26ayamrkfaij4st6amckemxzeau2evcihk2c5y55xzeqeyuca` |
| `VALID_API_KEYS` | API keys clients use to call our API | `ocid1.vaultsecret.oc1.us-chicago-1.amaaaaaakve26ayaaed4oxukns2lufos5cwkwpopez56b4eitmaenr2fml3a` |
| `SLACK_WEBHOOK_URL` | Webhook URL for Slack alerts | `ocid1.vaultsecret.oc1.us-chicago-1.amaaaaaakve26ayapsry5l7dg4ottdmdddf6dayvaoxtoggzrm4duk2gmmna` |
| `DD_API_KEY` | Datadog API key (placeholder — using Grafana stack instead) | `ocid1.vaultsecret.oc1.us-chicago-1.amaaaaaakve26ayabw43afqvw2dscibbxudy3tocsj2xsbcerbtkldqmn2wq` |
| `GROUND_TRUTH_INGEST_TOKEN` | Shared secret for POST /ground-truth/ingest | `ocid1.vaultsecret.oc1.us-chicago-1.amaaaaaakve26ayaln6kcqpxxxjrvnbuues24ssjfcle3l7d5wbtfh5xfhfa` |

**Why base64 encoding?**
Oracle Vault stores secrets as base64-encoded strings.
Base64 is not encryption — it just converts binary data to safe text
so it can travel over HTTP without corruption. The actual encryption
is done by the master key once it reaches Oracle's servers.

---

## Current State

```
Oracle Cloud Account (us-chicago-1)
└── Vault: nyc-airbnb-vault
    ├── Master Key: nyc-airbnb-master-key (AES-256, in Oracle HSM)
    ├── Secret: REDIS_PASSWORD      ✅
    ├── Secret: VALID_API_KEYS      ✅
    └── Secret: SLACK_WEBHOOK_URL   ✅
```

---

## What's Next

| Stage | What | Why |
|-------|------|-----|
| Stage 2 | Terraform | Write a blueprint to create the Kubernetes cluster with one command |
| Stage 3 | Docker multi-arch | Build an image that runs on Oracle's ARM servers |
| Stage 4 | Helm | Package the entire app as an installer — one command deploys everything |
| Stage 5 | Vault integration | Kubernetes pods read secrets from Vault at startup, no .env files ever |
