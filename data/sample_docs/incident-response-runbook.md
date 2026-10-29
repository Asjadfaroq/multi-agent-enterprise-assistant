---
title: Incident Response Runbook
acl: engineering
---
# Incident Response Runbook

This runbook describes how Product Engineering responds to production incidents in Ledgerly. It applies to every alert that pages the on-call engineer and to every P1 ticket escalated by Customer Success.

## Severity levels

- **SEV1**: Ledgerly is unavailable for all customers, payments are failing, or customer data may have been exposed. Response starts immediately, day or night.
- **SEV2**: a major feature is unavailable for many customers, or performance is degraded so that pages take longer than 5 seconds to load. Response starts within 30 minutes.
- **SEV3**: a minor feature is broken or a small group of customers is affected. Handled during working hours.

When in doubt, declare the higher severity. Downgrading later is cheap; reacting late to a real SEV1 is not.

## On-call rotation

The on-call rotation is one week long and changes every Sunday at 09:00 Riyadh time. There is always a primary and a secondary on-call engineer. The primary acknowledges a page within 10 minutes. If the primary does not acknowledge within 10 minutes, the page goes to the secondary, and after another 10 minutes to the VP of Engineering.

On-call engineers receive an allowance of 1,500 SAR per week on call, plus time off in lieu for any incident handled between 22:00 and 07:00.

## Declaring an incident

1. Acknowledge the page in the alerting tool.
2. Open a channel named `#inc-YYYYMMDD-short-name` and post the alert, the suspected impact and the severity.
3. For SEV1 and SEV2, assign an incident commander. The incident commander coordinates and communicates; they do not debug. The on-call engineer can be the commander only for SEV3.
4. For SEV1, the incident commander informs the support lead within 15 minutes so Customer Success can update the status page and reply to customers.

## During the incident

Post an update in the incident channel at least every 30 minutes for SEV1 and every hour for SEV2, even if nothing has changed. Updates state what we know, what we are trying, and when the next update will come.

Prefer mitigation over diagnosis. Rolling back the last deployment, disabling a feature flag, or scaling up the service is better than investigating a root cause while customers are affected. Rollback instructions are in the deployment process document.

Never run manual write queries against the production database during an incident without a second engineer reviewing the query in the incident channel.

## After the incident

The incident is resolved when customer impact has ended and monitoring is green for 30 minutes. The incident commander then:

- posts a short resolution message in the incident channel and #announcements,
- asks Customer Success to close the status page entry,
- schedules a blameless postmortem within 5 working days for SEV1 and SEV2.

The postmortem document lists the timeline, impact, root cause, what went well, what went badly, and action items with owners and due dates. Action items are tracked as tickets with the category `bug` and the label `postmortem`. The VP of Engineering reviews open postmortem action items every two weeks.

## Data exposure incidents

If customer data may have been exposed, treat the incident as SEV1 and inform the Head of Security and the CEO immediately. Do not discuss details in public channels. Regulators and affected customers are notified by Legal within 72 hours of confirming the exposure, as required by the Personal Data Protection Law.
