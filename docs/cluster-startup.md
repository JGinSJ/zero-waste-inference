# Cluster Startup Runbook

Definitive runbook for bringing the Akamai LKE cluster back up after a
scale-down or cold start. Execute sections in order from top to bottom.

**Cluster:** `akamai-lke-us-ord`  
**Namespace:** `inference`

---

## 1. Prerequisites

Before running any commands, confirm:

- `KUBECONFIG` is set:
  ```bash
  export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
  kubectl get nodes   # should show 3 nodes: 1 CPU + 2 GPU
  ```

- Docker Desktop is running (infernostra account) — required to pull images if
  any pod has `imagePullPolicy: Always` or is starting for the first time.

- Three terminal tabs will be needed for port-forwards in [Section 6](#6-port-forwards).
  Open them before starting if you plan to run benchmarks immediately after startup.

---

## 2. One-time setup steps

> **ONE-TIME — skip if already done.**  
> These steps are required once after initial cluster provisioning or after
> the cluster is destroyed and recreated via `terraform apply`. They are not
> needed for a routine scale-down/scale-up cycle.

### 2a. Install NVIDIA device plugin

Akamai LKE does **not** pre-install the NVIDIA device plugin. Without it,
`nvidia.com/gpu` resource requests are ignored and GPU pods will not schedule.

```bash
kubectl create -f https://raw.githubusercontent.com/NVIDIA/k8s-device-plugin/v0.17.3/deployments/static/nvidia-device-plugin.yml
```

Verify GPUs are advertised before proceeding:

```bash
kubectl -n kube-system get daemonset nvidia-device-plugin-daemonset
kubectl describe node lke591117-868011-613ea4520000 | grep nvidia.com/gpu
kubectl describe node lke591117-868011-4171fee90000 | grep nvidia.com/gpu
# Expected: nvidia.com/gpu: 1 on both GPU nodes
```

### 2b. Apply node labels

LKE does not propagate pool-level labels to Kubernetes Node objects. Apply
manually. These labels are the `nodeSelector` targets for all phase manifests.

```bash
# CPU node — required by Valkey and Fermyon (workload-type=cpu nodeSelector)
kubectl label node lke591117-865821-4a82b6ec0000 workload-type=cpu --overwrite

# GPU nodes — required by vLLM and Qwen-Image (gpu-type=rtx4000ada nodeSelector)
kubectl label node lke591117-868011-613ea4520000 gpu-type=rtx4000ada --overwrite
kubectl label node lke591117-868011-4171fee90000 gpu-type=rtx4000ada --overwrite
```

Verify:

```bash
kubectl get nodes --show-labels | grep -E "workload-type|gpu-type"
```

### 2c. Apply GPU taints

Prevents CPU-only pods from being scheduled on expensive GPU nodes.

```bash
kubectl taint nodes lke591117-868011-613ea4520000 gpu-type=rtx4000ada:NoSchedule
kubectl taint nodes lke591117-868011-4171fee90000 gpu-type=rtx4000ada:NoSchedule
```

Verify:

```bash
kubectl describe node lke591117-868011-613ea4520000 | grep -A2 Taints
kubectl describe node lke591117-868011-4171fee90000 | grep -A2 Taints
# Expected: gpu-type=rtx4000ada:NoSchedule
```

### 2d. Apply Fermyon deployment rename

**One-time migration — skip if already applied.** Renames the live Deployment
from `fermyon-proxy` to `fermyon-prefix-cache` to match the manifest, which now
uses `fermyon-prefix-cache` throughout (Deployment, labels, selector, container,
Service selector). Once done, the old Deployment no longer exists and the
`delete` below is a harmless no-op. Check first:

```bash
kubectl get deployment -n inference | grep fermyon
```

```bash
kubectl apply -f phases/phase2-prefix-cache/fermyon/k8s/fermyon-deployment.yaml
kubectl rollout status deployment/fermyon-prefix-cache -n inference
kubectl delete deployment fermyon-proxy -n inference
```

---

## 3. Scale up GPU workloads

Run after every scale-down. Valkey and Fermyon stay running and do not need
to be scaled.

```bash
kubectl scale deployment vllm -n inference --replicas=1
kubectl scale deployment qwen-image -n inference --replicas=1
```

---

## 4. Wait for readiness

GPU pods take 2–3 minutes to pull the model into VRAM and start serving.
Watch rollout status:

```bash
kubectl rollout status deployment/vllm -n inference
kubectl rollout status deployment/qwen-image -n inference
```

Or watch pod status directly:

```bash
kubectl get pods -n inference -w
# Wait until both vllm-* and qwen-image-* pods show STATUS = Running, READY = 1/1
```

Do not proceed to port-forwards until both pods are `Running 1/1`.

---

## 5. Re-apply updated manifests

> **Only needed when manifests have changed since the last `kubectl apply`.**  
> Skip this section for a routine scale-up with no manifest changes.

```bash
kubectl apply -f phases/phase2-prefix-cache/vllm/vllm.yaml
kubectl apply -f phases/phase3-qwen-image/k8s/deployment.yaml
kubectl apply -f phases/phase4-benchmarks/k8s/vllm-ada.yaml
```

---

## 6. Port-forwards

Run each in a **dedicated terminal tab** — do not background with `&` or the
port-forward will be killed when the session ends.

**Tab 1 — vLLM (Phase 2 backend, port 8000):**

```bash
export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
kubectl port-forward -n inference svc/vllm-svc 8000:8000
```

**Tab 2 — Fermyon prefix-cache (Phase 2 front door, port 8082):**

```bash
export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
kubectl port-forward -n inference svc/fermyon-svc 8082:8082
```

**Tab 3 — Qwen-Image (Phase 3, port 8080):**

The qwen-image pod name changes on every restart. Get the current name first:

```bash
kubectl get pods -n inference
# Look for the qwen-image-* pod in Running state, e.g.:
#   qwen-image-6b645d59c6-t5r82
```

Then port-forward using the pod name:

```bash
export KUBECONFIG=~/.kube/akamai-inference-lke.yaml
kubectl port-forward -n inference pod/<qwen-image-pod-name> 8080:8080
```

---

## 7. Verify services are healthy

Run after port-forwards are established.

**vLLM** (returns HTTP 200 with empty body — do not pipe to `json.tool`):

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8000/health
# Expected: 200
```

**Fermyon:**

```bash
curl -s -o /dev/null -w "%{http_code}" http://localhost:8082/health
# Expected: 200
# Body: ok
```

**Qwen-Image:**

```bash
curl -s http://localhost:8080/health
# Expected: 200
# Body: {"status":"ok","gpu":"NVIDIA RTX 4000 Ada Generation",
#        "model":"Qwen/Qwen2.5-VL-7B-Instruct","optimized":true,"dtype":"bfloat16"}
```

---

## 8. Scale down (when done)

Scales GPU workloads to zero replicas. Valkey and Fermyon remain running.

```bash
kubectl scale deployment vllm -n inference --replicas=0
kubectl scale deployment qwen-image -n inference --replicas=0
```

GPU pods stop running and free their VRAM, but GPU nodes continue to be billed
until the node pool is removed. To stop GPU billing entirely, see Section 9.

---

## 9. Cost reference

| Pool | Plan | Nodes | Rate | When charged |
|---|---|---|---|---|
| CPU (865821) | g6-dedicated-4 | 1 | $0.108/hr | Always |
| GPU (868011) | g2-gpu-rtx4000a1-l (RTX 4000 Ada) | 2 | $1.920/hr | Always (node exists) |
| **Total with GPU nodes** | | | **$2.028/hr** | |
| **Total GPU scaled to 0** | | | **$0.108/hr** | CPU only |

> GPU nodes are billed for as long as they exist in the pool, regardless of
> pod replica count. Scaling deployments to 0 stops GPU workloads but does
> not remove the nodes. To stop GPU billing entirely, remove the node pool via
> `terraform apply` with `ada_node_count = 0` — but this will destroy the nodes
> and require the one-time setup steps in Section 2 to be repeated on next
> provisioning.
