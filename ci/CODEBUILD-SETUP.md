# AWS CodeBuild setup for `vllm-deepep-v2-efa`

This document is the one-time provisioning recipe for the AWS CodeBuild
project that builds, preflights, and pushes this image to ECR.

Everything here is a template - replace the placeholder account ID,
region, GitHub PAT, and role names with values from your AWS account.
No account IDs or secrets are hardcoded into any checked-in file.

## What the pipeline does

1. Pulls the GHCR base image
   `ghcr.io/antonai-work/deepep-v2-efa-base:v0.1.0-sm90a` (needs either
   a public GHCR package or a GitHub PAT with `read:packages` scope in
   Secrets Manager).
2. Runs `docker build --build-arg BUILD_MODE=fast`, then runs the
   in-image preflight (`/opt/docker/preflight.sh`). Requires `8/8
   checks PASS`. Exits non-zero on any failure.
3. Runs `docker build --build-arg BUILD_MODE=vanilla` (validation-only;
   proves the repo is offline-reproducible from vanilla `nvidia/cuda`).
   Runs the same preflight.
4. Pushes the fast build to ECR as
   `<account>.dkr.ecr.<region>.amazonaws.com/vllm-deepep-v2-efa:fast-<git-sha>`
   plus `:fast-latest`. Vanilla build is never pushed.

See `ci/buildspec.yml` for the exact commands; this doc only covers AWS
resource provisioning.

## Prerequisites

- AWS account with ECR, CodeBuild, IAM, Secrets Manager access.
- GitHub PAT with `read:packages` scope (only if the base GHCR package
  stays private). Create one at
  <https://github.com/settings/tokens> and name it something like
  `ghcr-read-packages-codebuild`.
- Local AWS CLI authenticated to the target account.

Export placeholders once so all commands below Just Work:

```bash
export AWS_ACCOUNT_ID=123456789012          # replace
export AWS_REGION=us-east-2                 # replace
export ECR_REPO=vllm-deepep-v2-efa
export CB_PROJECT=vllm-deepep-v2-efa-ci
export CB_ROLE_NAME=vllm-deepep-v2-efa-codebuild-role
export GHCR_SECRET_NAME=ghcr-read-token-vllm-deepep-v2-efa
export GHCR_USER=antonai-work                # GitHub login/org that owns the PAT
export GH_REPO=antonai-work/vllm-deepep-v2-efa
```

## 1. Create the ECR repository

```bash
aws ecr create-repository \
    --region "${AWS_REGION}" \
    --repository-name "${ECR_REPO}" \
    --image-scanning-configuration scanOnPush=true \
    --image-tag-mutability MUTABLE
```

Note the returned `repositoryUri`; the buildspec composes it from
`${AWS_ACCOUNT_ID}.dkr.ecr.${AWS_REGION}.amazonaws.com/${ECR_REPO}`.

## 2. Store the GHCR read token in Secrets Manager

Skip this section if you flip the base GHCR package to public. The
buildspec checks for an empty `GHCR_READ_TOKEN_SECRET_ARN` and silently
falls back to anonymous pulls in that case.

```bash
# Paste a PAT with read:packages scope, or use --secret-string file:///path.
aws secretsmanager create-secret \
    --region "${AWS_REGION}" \
    --name "${GHCR_SECRET_NAME}" \
    --description "GitHub PAT with read:packages for GHCR base image" \
    --secret-string "ghp_PASTE_A_REAL_TOKEN_HERE"

GHCR_SECRET_ARN=$(aws secretsmanager describe-secret \
    --region "${AWS_REGION}" \
    --secret-id "${GHCR_SECRET_NAME}" \
    --query ARN --output text)
echo "GHCR_SECRET_ARN=${GHCR_SECRET_ARN}"
```

## 3. Create the CodeBuild IAM role

Trust policy (lets CodeBuild assume the role):

```bash
cat >/tmp/codebuild-trust.json <<'EOF'
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Effect": "Allow",
      "Principal": { "Service": "codebuild.amazonaws.com" },
      "Action": "sts:AssumeRole"
    }
  ]
}
EOF

aws iam create-role \
    --role-name "${CB_ROLE_NAME}" \
    --assume-role-policy-document file:///tmp/codebuild-trust.json
```

Inline permission policy (CloudWatch Logs, ECR, Secrets Manager read):

```bash
cat >/tmp/codebuild-policy.json <<EOF
{
  "Version": "2012-10-17",
  "Statement": [
    {
      "Sid": "CloudWatchLogs",
      "Effect": "Allow",
      "Action": [
        "logs:CreateLogGroup",
        "logs:CreateLogStream",
        "logs:PutLogEvents"
      ],
      "Resource": "arn:aws:logs:${AWS_REGION}:${AWS_ACCOUNT_ID}:log-group:/aws/codebuild/${CB_PROJECT}*"
    },
    {
      "Sid": "ECRAuth",
      "Effect": "Allow",
      "Action": "ecr:GetAuthorizationToken",
      "Resource": "*"
    },
    {
      "Sid": "ECRRepoPushPull",
      "Effect": "Allow",
      "Action": [
        "ecr:BatchCheckLayerAvailability",
        "ecr:GetDownloadUrlForLayer",
        "ecr:BatchGetImage",
        "ecr:PutImage",
        "ecr:InitiateLayerUpload",
        "ecr:UploadLayerPart",
        "ecr:CompleteLayerUpload"
      ],
      "Resource": "arn:aws:ecr:${AWS_REGION}:${AWS_ACCOUNT_ID}:repository/${ECR_REPO}"
    },
    {
      "Sid": "SecretsManagerRead",
      "Effect": "Allow",
      "Action": "secretsmanager:GetSecretValue",
      "Resource": "arn:aws:secretsmanager:${AWS_REGION}:${AWS_ACCOUNT_ID}:secret:${GHCR_SECRET_NAME}-*"
    }
  ]
}
EOF

aws iam put-role-policy \
    --role-name "${CB_ROLE_NAME}" \
    --policy-name "${CB_PROJECT}-inline" \
    --policy-document file:///tmp/codebuild-policy.json

CB_ROLE_ARN="arn:aws:iam::${AWS_ACCOUNT_ID}:role/${CB_ROLE_NAME}"
echo "CB_ROLE_ARN=${CB_ROLE_ARN}"
```

If CodeBuild needs to pull the source from a private GitHub repo,
additionally attach CodeBuild's GitHub source credentials or connect
through AWS CodeStar Connections (out of scope of this doc; the repo
itself is public).

## 4. Create the CodeBuild project

```bash
cat >/tmp/codebuild-project.json <<EOF
{
  "name": "${CB_PROJECT}",
  "description": "Dual-path docker build + preflight + ECR push for vllm-deepep-v2-efa",
  "source": {
    "type": "GITHUB",
    "location": "https://github.com/${GH_REPO}.git",
    "buildspec": "ci/buildspec.yml",
    "reportBuildStatus": true
  },
  "sourceVersion": "main",
  "artifacts": { "type": "NO_ARTIFACTS" },
  "environment": {
    "type": "LINUX_CONTAINER",
    "image": "aws/codebuild/amazonlinux2-x86_64-standard:5.0",
    "computeType": "BUILD_GENERAL1_2XLARGE",
    "privilegedMode": true,
    "environmentVariables": [
      { "name": "AWS_ACCOUNT_ID",            "value": "${AWS_ACCOUNT_ID}",    "type": "PLAINTEXT" },
      { "name": "AWS_REGION",                "value": "${AWS_REGION}",        "type": "PLAINTEXT" },
      { "name": "ECR_REPO",                  "value": "${ECR_REPO}",          "type": "PLAINTEXT" },
      { "name": "GHCR_USER",                 "value": "${GHCR_USER}",         "type": "PLAINTEXT" },
      { "name": "GHCR_READ_TOKEN_SECRET_ARN","value": "${GHCR_SECRET_ARN}",   "type": "PLAINTEXT" }
    ]
  },
  "serviceRole": "${CB_ROLE_ARN}",
  "timeoutInMinutes": 90,
  "queuedTimeoutInMinutes": 60
}
EOF

aws codebuild create-project \
    --region "${AWS_REGION}" \
    --cli-input-json file:///tmp/codebuild-project.json
```

Compute choice: `BUILD_GENERAL1_2XLARGE` (72 vCPU, 145 GB RAM). The
cold vanilla build spends most of its time compiling DeepEP V2 and
aws-ofi-nccl; more cores is more throughput. The fast path barely
needs it but keeps the two phases symmetric.

`privilegedMode=true` is required for `docker build` inside CodeBuild.

## 5. Trigger a build

```bash
aws codebuild start-build \
    --region "${AWS_REGION}" \
    --project-name "${CB_PROJECT}" \
    --source-version main
```

Watch logs:

```bash
BUILD_ID=$(aws codebuild list-builds-for-project \
    --region "${AWS_REGION}" \
    --project-name "${CB_PROJECT}" \
    --query 'ids[0]' --output text)
aws codebuild batch-get-builds --region "${AWS_REGION}" --ids "${BUILD_ID}" \
    | jq '.builds[0].logs'
```

## 6. Wire webhooks (optional)

```bash
aws codebuild create-webhook \
    --region "${AWS_REGION}" \
    --project-name "${CB_PROJECT}" \
    --filter-groups '[[{"type":"EVENT","pattern":"PUSH"},{"type":"HEAD_REF","pattern":"^refs/heads/main$"}]]'
```

## Troubleshooting

- **GHCR pull fails with `denied`:** the PAT does not have
  `read:packages`, or the Secrets Manager value is stale. Recreate the
  secret:
  `aws secretsmanager put-secret-value --secret-id "${GHCR_SECRET_NAME}" --secret-string "ghp_NEW_TOKEN"`
- **Preflight fails with NCCL < 2.30.4:** the build likely pulled a
  stale base layer. Clear the BuildKit cache: `aws codebuild
  update-project --name "${CB_PROJECT}" --cache '{"type":"NO_CACHE"}'`.
- **Out of disk:** the CodeBuild worker has a default 64 GB ephemeral
  volume; both builds together can exceed this. Switch to a larger
  `computeType` (`BUILD_GENERAL1_LARGE_DISK` family) or prune between
  phases (`docker image prune -f`).
- **`docker: command not found`:** `privilegedMode` is false. Flip it
  via `aws codebuild update-project --environment privilegedMode=true`.

## Sibling repos with equivalent pipelines

- [`antonai-work/deepep-v2-efa-base`](https://github.com/antonai-work/deepep-v2-efa-base) -
  publishes the base image this pipeline's fast path pulls.
- [`antonai-work/nemo-rl-deepep-v2-efa`](https://github.com/antonai-work/nemo-rl-deepep-v2-efa) -
  same dual-path pattern for Megatron-LM + NeMo-RL training.
