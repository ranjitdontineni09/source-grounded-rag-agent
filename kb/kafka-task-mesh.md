# Kafka task mesh

Submit a job over REST. Kafka workers run it in parallel. PostgreSQL stores status. A light UI tracks the job from submit to done or dead-letter.

Stack: Java 17, Spring Boot, Kafka, Docker, PostgreSQL.

At-least-once delivery is safe because workers claim a row with a conditional update before they run it. Retryable failures back off. Poison payloads go to a dead-letter topic. A lease reaper re-queues workers that crashed mid-job.

Finish is owner CAS: complete and fail only succeed when status is RUNNING and worker_id matches. Exhausted attempts become FAILED. Manual retry re-enqueues FAILED or DEAD_LETTER.
