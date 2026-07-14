# Meridian Analytics — Security & Compliance

## Encryption

All data is encrypted in transit using TLS 1.2 or higher. Data at rest is encrypted
with AES-256. Encryption keys are managed by Meridian's KMS and rotated every 90 days.

## Authentication

The Console supports email/password login and SAML single sign-on. SAML SSO is available
on the Team and Enterprise editions only. Multi-factor authentication (MFA) is available
on all editions and is enforced by default on Enterprise.

## API keys

Query API access uses bearer tokens. Tokens are scoped to a single workspace and can be
granted `read` or `read-write` scope. Tokens do not expire automatically; an administrator
must revoke them manually. Meridian recommends rotating API tokens every 180 days.

## Compliance

Meridian Systems Inc. is SOC 2 Type II certified. It is **not** HIPAA compliant, and
customers must not upload protected health information (PHI) to the platform. A GDPR data
processing addendum (DPA) is available for EU customers on request.

## Incident response

Security incidents can be reported to `security@meridian.example`. Meridian commits to
acknowledging reports within 24 hours. Confirmed critical vulnerabilities are eligible
for a bug-bounty reward of up to $5,000.
