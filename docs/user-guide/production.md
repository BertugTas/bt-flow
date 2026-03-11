# Production Deployment

## Gunicorn + Uvicorn workers

For multi-worker production deployments:

```bash
gunicorn -k uvicorn.workers.UvicornWorker \
  -w 4 \
  -b 0.0.0.0:8000 \
  myapp:api.app
```

Where `myapp.py` contains:

```python
from bt import APIGenerator
api = APIGenerator("model.joblib", title="Production API")
```

## Docker

```dockerfile
FROM python:3.12-slim

WORKDIR /app
COPY model.joblib .
RUN pip install --no-cache-dir bt-flow

EXPOSE 8000
CMD ["bt-flow", "serve", "model.joblib", "--host", "0.0.0.0", "--port", "8000"]
```

## Kubernetes

Use `/health` as a liveness/readiness probe:

```yaml
livenessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 5
  periodSeconds: 10

readinessProbe:
  httpGet:
    path: /health
    port: 8000
  initialDelaySeconds: 3
  periodSeconds: 5
```

## Security considerations

!!! warning
    bt-flow provides **no built-in authentication**. Before exposing to the internet:
    
    - Add an API gateway with authentication (e.g., AWS API Gateway, Kong)
    - Use HTTPS (TLS termination via nginx/Caddy/load balancer)
    - Restrict access with network policies or firewall rules

!!! danger
    Never load model files from untrusted sources. `.pkl` / `.joblib` files can execute arbitrary code during deserialisation.
