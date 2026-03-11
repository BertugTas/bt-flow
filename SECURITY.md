# Security Policy

## Supported Versions

| Version | Supported          |
|---------|--------------------|
| 0.1.x   | ✅ Active support  |

---

## Reporting a Vulnerability

**Please do not report security vulnerabilities via public GitHub Issues.**

To report a vulnerability, email **bertugtas@example.com** with:

- A description of the vulnerability and its potential impact
- Steps to reproduce (proof of concept if available)
- Any suggested mitigations

You will receive an acknowledgement within **48 hours** and a patch timeline within **7 days**.

---

## Scope

The following are **in scope**:

- Remote code execution via crafted model files (pickle deserialisation)
- Authentication or authorisation bypass in generated APIs
- Dependency vulnerabilities (please also report upstream)

The following are **out of scope**:

- Vulnerabilities in the user's model logic
- Issues caused by running bt-flow with untrusted models (user responsibility)

---

## Security Notes

> [!CAUTION]
> bt-flow uses `joblib.load` / `pickle.load` to deserialise model files. **Never load model files from untrusted sources.** A maliciously crafted `.pkl` file can execute arbitrary code.

- bt-flow generated APIs have **no built-in authentication**. Add an API gateway, OAuth2 middleware, or network-level controls before exposing endpoints to the public internet.
- Validate and sanitise input data at the infrastructure level for production deployments.
