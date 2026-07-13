# Security Policy

## Supported versions

This is an educational lab, developed against the `main` branch. Security
fixes land on `main` only.

| Version | Supported |
|---------|-----------|
| `main`  | ✅        |

## Threat surface

llm-ecosphere is a local, offline lab:

- **No runtime network egress.** Training, evaluation, sampling and
  interactive play never open a network connection. The only network
  activity is `pip`/`uv` fetching dependencies at setup time.
- **No secrets, no credentials, no user data.** The entire "world" is a
  deterministic toy game; every artifact under `data/` and `runs/` is
  reproducible from the code.
- **Dependencies**: `torch`, `numpy`, `pytest` (see `requirements.txt`).
  The CI toolchain (MkDocs, Semgrep) is hash-pinned in
  `requirements/*.lock.txt`.

## Supply-chain controls

- Every GitHub Actions `uses:` reference is pinned to a 40-hex commit SHA,
  enforced by a CI gate on every PR.
- The docs and SAST toolchains install with `pip --require-hashes` against
  generated lockfiles.
- gitleaks scans every push for inadvertently committed secrets; Semgrep
  (`p/security-audit` + `p/secrets`) runs as a blocking check.
- Dependabot keeps actions and Python dependencies current.
- OpenSSF Scorecard runs weekly and publishes to
  api.securityscorecards.dev.

## Reporting a vulnerability

Please do **not** open a public issue for security-relevant findings.
Use GitHub's private vulnerability reporting instead:

→ <https://github.com/yves-vogl/llm-ecosphere/security/advisories/new>

You will get an initial response within 7 days. Please include a minimal
reproduction and the commit hash you tested against.
