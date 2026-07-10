#!/usr/bin/env bash
# One-time AWS setup for the public caffeine status feed (see status_publisher.py).
#
# Run this wherever you have AWS admin credentials (laptop with aws CLI, or
# AWS CloudShell in the console — CloudShell needs no local setup):
#
#   ./setup-aws.sh <bucket-name> [region]
#   e.g. ./setup-aws.sh aarg-status us-west-2
#
# It creates: the bucket, a public-read policy for the one feed object, a CORS
# rule allowing cross-origin GET from anywhere, and an IAM user whose only
# permission is PutObject on that one key. It prints the .env lines for the
# Pi 5 and the public URL for the website's STATUS_URL.
#
# PRIVACY: the feed object (desk presence + caffeine log) is world-readable.
set -euo pipefail

BUCKET="${1:?usage: ./setup-aws.sh <bucket-name> [region]}"
REGION="${2:-us-west-2}"
KEY="caffeine.json"
IAM_USER="${BUCKET}-publisher"

echo "== Creating bucket s3://$BUCKET in $REGION =="
if [ "$REGION" = "us-east-1" ]; then
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION"
else
  aws s3api create-bucket --bucket "$BUCKET" --region "$REGION" \
    --create-bucket-configuration "LocationConstraint=$REGION"
fi

echo "== Allowing a public bucket policy (ACLs stay blocked) =="
aws s3api put-public-access-block --bucket "$BUCKET" \
  --public-access-block-configuration \
  "BlockPublicAcls=true,IgnorePublicAcls=true,BlockPublicPolicy=false,RestrictPublicBuckets=false"

echo "== Public read policy scoped to $KEY only =="
aws s3api put-bucket-policy --bucket "$BUCKET" --policy "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Sid\": \"PublicReadStatusFeed\",
    \"Effect\": \"Allow\",
    \"Principal\": \"*\",
    \"Action\": \"s3:GetObject\",
    \"Resource\": \"arn:aws:s3:::$BUCKET/$KEY\"
  }]
}"

echo "== CORS: allow cross-origin GET/HEAD from anywhere =="
aws s3api put-bucket-cors --bucket "$BUCKET" --cors-configuration "{
  \"CORSRules\": [{
    \"AllowedOrigins\": [\"*\"],
    \"AllowedMethods\": [\"GET\", \"HEAD\"],
    \"AllowedHeaders\": [\"*\"],
    \"MaxAgeSeconds\": 300
  }]
}"

echo "== IAM user $IAM_USER with PutObject on that one key =="
aws iam create-user --user-name "$IAM_USER" >/dev/null
aws iam put-user-policy --user-name "$IAM_USER" \
  --policy-name "put-${BUCKET}-caffeine-json" \
  --policy-document "{
  \"Version\": \"2012-10-17\",
  \"Statement\": [{
    \"Effect\": \"Allow\",
    \"Action\": \"s3:PutObject\",
    \"Resource\": \"arn:aws:s3:::$BUCKET/$KEY\"
  }]
}"

CREDS=$(aws iam create-access-key --user-name "$IAM_USER" \
  --query 'AccessKey.[AccessKeyId,SecretAccessKey]' --output text)
ACCESS_KEY=$(echo "$CREDS" | cut -f1)
SECRET_KEY=$(echo "$CREDS" | cut -f2)

URL="https://${BUCKET}.s3.${REGION}.amazonaws.com/${KEY}"

echo
echo "== DONE. Add these lines to the Pi 5's ~/ai_desk_agent/.env =="
echo
echo "AWS_ACCESS_KEY_ID=$ACCESS_KEY"
echo "AWS_SECRET_ACCESS_KEY=$SECRET_KEY"
echo "AWS_DEFAULT_REGION=$REGION"
echo "STATUS_S3_BUCKET=$BUCKET"
echo
echo "Public feed URL (the website's STATUS_URL):"
echo "  $URL"
echo
echo "Verify after restarting the ai-eink service:"
echo "  curl -s $URL"
echo "  curl -s -H 'Origin: https://aarg.dev' -D - -o /dev/null $URL | grep -i access-control"
