# Build on top of v9 which has Llama 3.1 model + all Python deps already installed
FROM sabya610/rag-app:llama3.1-8b-q4-v9

# Override only the app code with the latest version (SFDC integration + perf fixes)
COPY rag_app/app /app/rag_app/app


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
