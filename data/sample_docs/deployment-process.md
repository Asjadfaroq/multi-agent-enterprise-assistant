---
title: Deployment Process
acl: engineering
---
# Deployment Process

## Branches and reviews

All changes go through a pull request to `main`. A pull request needs one approving review and a green CI run before it can be merged. Changes to billing or payments code need two approving reviews, one from the payments team.

## Releases

Merging to `main` deploys automatically to staging. Production deploys run from the release pipeline on Sunday to Wednesday between 10:00 and 15:00 Riyadh time. There are no production deploys on Thursdays, Fridays, Saturdays, or the day before a public holiday, unless it is a hotfix for a SEV1 or SEV2 incident.

Every production deploy uses a canary: 10% of traffic for 15 minutes, then 100% if the error rate stays below 1%.

## Feature flags

New customer-facing features ship behind a feature flag. Flags are removed within 30 days of reaching 100% rollout.

## Rollback

To roll back, run the release pipeline with the previous release tag and the `rollback` option. A rollback skips the canary and takes about 5 minutes. Database migrations must be backwards compatible with the previous release so that a rollback never needs a schema change.
