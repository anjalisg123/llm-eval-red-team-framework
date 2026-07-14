# Meridian Analytics — Product Overview

Meridian Analytics is a cloud data-analytics platform operated by Meridian Systems Inc.
It ingests event streams, stores them in a columnar warehouse, and exposes dashboards
and a query API.

## Editions

Meridian ships in three editions:

- **Starter** — up to 5 million events per month, 3 seats, 7-day data retention.
- **Team** — up to 50 million events per month, 25 seats, 90-day data retention.
- **Enterprise** — unlimited events, unlimited seats, 365-day data retention,
  and a 99.95% uptime SLA.

Only the Enterprise edition includes a contractual uptime SLA. Starter and Team are
offered on a best-effort basis with no SLA.

## Core components

- **Collector** — receives events over HTTPS at `https://ingest.meridian.example`.
- **Warehouse** — a columnar store; data is partitioned by day.
- **Query API** — a REST API at `https://api.meridian.example/v2/query`.
- **Console** — the web dashboard UI.

## Regions

Meridian runs in three regions: `us-east`, `eu-west`, and `ap-south`. Data does not
move between regions. A workspace is pinned to a single region when it is created and
cannot be migrated afterward without opening a support ticket.
