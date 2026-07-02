# Build on top of v14 which has Llama 3.1 + SFDC + perf fixes
FROM sabya610/rag-app:llama3.1-8b-q4-v14

# Install Slack SDK
RUN pip install --no-cache-dir slack-sdk==3.34.0

# SFDC-only mode (skip LLM) + Slack integration
COPY rag_app/app/routes/rag_routes.py /app/rag_app/app/routes/rag_routes.py
COPY rag_app/app/routes/slack_routes.py /app/rag_app/app/routes/slack_routes.py
COPY rag_app/app/services/slack_service.py /app/rag_app/app/services/slack_service.py
COPY rag_app/app/config.py /app/rag_app/app/config.py
COPY rag_app/app/__init__.py /app/rag_app/app/__init__.py


# Set environment variables
ENV PYTHONPATH=/app/rag_app \
    PYTHONUNBUFFERED=1 \
    FLASK_APP=run.py \
    TRANSFORMERS_OFFLINE=1


#RUN python -c "from sentence_transformers import SentenceTransformer; \
#    SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2').save('/app/rag_app/app/models/embedding')"


# Expose Flask/Gunicorn port
EXPOSE 5000

# Run app with Gunicorn — timeout 1800s for slow LLM inference, no --preload to avoid startup DB issues
CMD ["gunicorn", "-w", "1", "-b", "0.0.0.0:5000", "--timeout", "1800", "rag_app.run:app"]
