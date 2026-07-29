# deploy/

- **`example-images/`** — bundled first-run sample input (15 rock/paper/scissors
  photos, one directory per class; see `example-images/LICENSE.DATA` for the
  data license). In containerized mode, `docker compose up` bind-mounts this
  directory to `/data/images`, which is the default filesystem browse root — a
  clean clone can run the whole labeling loop against it with no external
  dataset. Local-source mode opens at `/` by default; set `IMAGE_ROOT` only when
  you want the picker contained to this directory (see `config.yaml.example`).

Deployment itself is documented in the [deployment guide](../docs/deployment.md)
and the repository [README](../README.md) quick starts.
