FROM node:20-alpine AS ui
WORKDIR /ui
COPY frontend/package.json ./
RUN npm install
COPY frontend/ ./
RUN npm run build

FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt
COPY agent ./agent
COPY kb ./kb
COPY lora ./lora
COPY --from=ui /ui/dist ./frontend/dist
ENV PYTHONUNBUFFERED=1
EXPOSE 8000
CMD ["uvicorn", "agent.app:app", "--host", "0.0.0.0", "--port", "8000"]
