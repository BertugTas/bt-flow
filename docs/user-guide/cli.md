# CLI Reference

## `bt-flow serve`

```
bt-flow serve [OPTIONS] MODEL_PATH
```

Serve a serialised scikit-learn model as a FastAPI REST API.

### Arguments

| Argument     | Description                                    |
|--------------|------------------------------------------------|
| `MODEL_PATH` | Path to `.pkl` or `.joblib` model file         |

### Options

| Option                   | Default          | Description                                              |
|--------------------------|------------------|----------------------------------------------------------|
| `--host`, `-H`           | `0.0.0.0`        | Network interface to bind to                             |
| `--port`, `-p`           | `8000`           | TCP port (1–65535)                                       |
| `--title`, `-t`          | `bt-flow Model API` | Title shown in Swagger UI                             |
| `--log-level`, `-l`      | `info`           | Uvicorn log level: `debug \| info \| warning \| error \| critical` |
| `--feature-names`, `-f`  | *(from model)*   | Comma-separated feature names, e.g. `'age,income,score'` |
| `--version`, `-V`        | —                | Print version and exit                                   |
| `--help`                 | —                | Show help and exit                                       |

### Examples

```bash
# Minimal
bt-flow serve iris_classifier.joblib

# Custom port and host
bt-flow serve model.pkl --port 9000 --host 127.0.0.1

# Override feature names
bt-flow serve model.pkl --feature-names 'sepal_len,sepal_wid,petal_len,petal_wid'

# Debug logging
bt-flow serve model.pkl --log-level debug --title "Debug API"
```
