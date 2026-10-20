# Security Policy

## Reporting a vulnerability

Please do not open a public issue for security problems. Report them privately through
[GitHub private vulnerability reporting](https://github.com/Monem-Benjeddou/azure-agents-starter/security/advisories/new).
You will get a response within a few days.

## Supported versions

Only the latest commit on `main` receives fixes.

## Handling secrets

Never commit `.env` files or keys. Secret scanning with push protection is enabled on this
repository. Prefer Microsoft Entra ID over API keys where possible; see the README section
on Azure OpenAI authentication.
