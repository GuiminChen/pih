# Security Policy

## Supported versions

PIH is currently a pre-stable developer preview. Security fixes are applied to
the latest revision on the default branch; older revisions are not supported.

## Reporting a vulnerability

Do not open a public issue for a suspected vulnerability. Use GitHub's private
vulnerability reporting feature for this repository. Include the affected
revision, deployment assumptions, reproduction steps and expected impact.

If private vulnerability reporting is not enabled, contact the repository owner
privately through their GitHub profile and ask for a secure reporting channel.
Do not include exploit details in that initial message.

PIH has not yet made production support claims for any GPU/model profile.
Nevertheless, issues involving artifact integrity, unsafe native loading,
capability confusion, memory safety, request isolation or remote interfaces are
treated as security-sensitive.
