FROM --platform=linux/amd64 pytorch/pytorch:2.9.1-cuda12.6-cudnn9-runtime AS example_algorithm_amd64
# Use a 'large' base container to show-case how to load pytorch and use the GPU (when enabled)
# Ensures that Python output to stdout/stderr is not buffered: prevents missing information when terminating
ENV PYTHONUNBUFFERED=1

RUN groupadd -r user && useradd -m --no-log-init -r -g user user
USER user

WORKDIR /opt/app

COPY --chown=user:user *.py /opt/app/
COPY --chown=user:user requirements.txt /opt/app/

# You can add any Python dependencies to requirements.txt
RUN python -m pip install \
    --user \
    --no-cache-dir \
    --no-color \
    --requirement /opt/app/requirements.txt

# Pre-compile into bytecode to import faster
# RUN python -m compileall /opt/conda/lib/python3.11/site-packages
RUN python -m compileall /home/user/.local/lib/python3.11/site-packages
RUN python -m compileall /opt/app/

LABEL org.grand-challenge.api-method="invoke"

ENTRYPOINT ["python", "app.py"]
