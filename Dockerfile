FROM python:3.12-slim
WORKDIR /app
ENV PYTHONDONTWRITEBYTECODE=1 PYTHONUNBUFFERED=1
COPY pyproject.toml ./
COPY shopping_agent ./shopping_agent
COPY fixtures ./fixtures
ARG INSTALL_VISION=0
RUN if [ "$INSTALL_VISION" = "1" ]; then pip install --no-cache-dir '.[vision]'; else pip install --no-cache-dir .; fi
ENTRYPOINT ["python", "-m", "shopping_agent"]
