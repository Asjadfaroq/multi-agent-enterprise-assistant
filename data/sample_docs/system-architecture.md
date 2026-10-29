---
title: Ledgerly System Architecture
acl: engineering
---
# Ledgerly System Architecture

## Services

- **web**: the customer-facing Next.js application.
- **api**: the main Python (FastAPI) service. It owns invoices, customers and users.
- **payments**: a separate service that talks to the payment gateways. Only this service holds gateway credentials.
- **einvoice**: signs invoices and submits them to ZATCA.
- **worker**: runs background jobs such as reminder emails and exports.

## Data

The primary database is PostgreSQL 16 with one primary and two read replicas. Backups are taken every night and kept for 35 days; point-in-time recovery is available for the last 7 days. Redis is used for caching and the job queue.

## Hosting

All services run on Kubernetes in a cloud region inside Saudi Arabia so customer data stays in the country. Staging runs in the same region on a smaller cluster.

## Limits

The public API allows 600 requests per minute per customer account. Requests above the limit receive HTTP 429 with a `Retry-After` header. Enterprise customers can ask for a higher limit through their account manager.
