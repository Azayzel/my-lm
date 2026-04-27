# Security Policy

## Supported versions

My-LM is pre-1.0 and ships from `main`. Only the latest commit on `main` is supported. Please update before reporting bugs.

## Reporting a vulnerability

**Do not open a public GitHub issue for security vulnerabilities.**

Email **contact@lavely.io** with:

- A description of the vulnerability
- Steps to reproduce, or a proof-of-concept
- The impact you observed (data exposure, RCE, credential leakage, etc.)
- Your suggested mitigation, if any

You'll get an acknowledgment within 5 business days. We aim to triage within 14 days and ship a fix on a timeline proportional to severity.

## Scope

In scope:

- The Python bridges in `scripts/` (RCE, command injection, path traversal, deserialization)
- The Electron main/preload (`contextBridge` API surface, IPC handlers, file system access)
- Default configuration that exposes secrets or weakens isolation
- Dependency vulnerabilities with a clear exploit path through My-LM

Out of scope:

- Vulnerabilities in third-party model weights
- Issues that require physical access to an unlocked machine
- Theoretical attacks without a practical exploit
- Self-XSS via the user's own prompts (renderer is sandboxed and `nodeIntegration: false`)

## Disclosure

We follow coordinated disclosure. After a fix is released we will credit the reporter (unless anonymity is requested) in the release notes.
