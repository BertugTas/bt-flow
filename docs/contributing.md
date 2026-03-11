# Contributing

See the [CONTRIBUTING.md](https://github.com/BertugTas/bt-flow/blob/main/CONTRIBUTING.md) file in the repository for the full contribution guide.

## Quick reference

```bash
# Clone and set up
git clone https://github.com/BertugTas/bt-flow.git
cd bt-flow
pip install hatch
pip install -e ".[dev]"

# Run everything
hatch run all     # lint + typecheck + test

# Or individually
hatch run test
hatch run lint
hatch run typecheck
```
