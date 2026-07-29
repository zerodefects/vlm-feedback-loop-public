# Contributing Guidelines

We're posting these examples on GitHub to support the NVIDIA LLM community and facilitate feedback.
We invite contributions!

Use the following guidelines to contribute to this project.


## Pull Requests
Developer workflow for code contributions is as follows:

1. Developers must first [fork](https://docs.github.com/en/pull-requests/collaborating-with-pull-requests/working-with-forks/fork-a-repo) this repository.
2. Git clone the forked repository and push changes to the personal fork.
3. Once the code changes are staged on the fork and ready for review, a Pull Request (PR) can be requested to merge the changes from a branch of the fork into a selected branch of upstream.
4. CI (`.github/workflows/ci.yml`) runs backend lint, backend tests, strict pyright type checking, frontend lint, frontend tests, frontend build, a dependency audit (pip-audit + pnpm audit), and a compose smoke (containerized build + health check) on every PR, plus SonarQube analysis (`.github/workflows/sonarqube.yml`); pre-commit hooks include ruff and gitleaks secret scanning. All checks must pass, and the PR is accepted after review by the repository owners.


## Signing Your Work
We require that all contributors "sign-off" on their commits. This certifies that the contribution is your original work, or you have rights to submit it under the same license, or a compatible license.

Any contribution which contains commits that are not Signed-Off will not be accepted.
To sign off on a commit, use the `--signoff` (or `-s`) option when committing your changes:

`$ git commit -s -m "Add cool feature."`
This will append the following to your commit message:

Signed-off-by: Your Name <your@email.com>


## Developer Certificate of Origin
Version 1.1

Copyright (C) 2004, 2006 The Linux Foundation and its contributors.

Everyone is permitted to copy and distribute verbatim copies of this
license document, but changing it is not allowed.


Developer's Certificate of Origin 1.1

By making a contribution to this project, I certify that:

(a) The contribution was created in whole or in part by me and I
    have the right to submit it under the open source license
    indicated in the file; or

(b) The contribution is based upon previous work that, to the best
    of my knowledge, is covered under an appropriate open source
    license and I have the right under that license to submit that
    work with modifications, whether created in whole or in part
    by me, under the same open source license (unless I am
    permitted to submit under a different license), as indicated
    in the file; or

(c) The contribution was provided directly to me by some other
    person who certified (a), (b) or (c) and I have not modified
    it.

(d) I understand and agree that this project and the contribution
    are public and that a record of the contribution (including all
    personal information I submit with it, including my sign-off) is
    maintained indefinitely and may be redistributed consistent with
    this project or the open source license(s) involved.
