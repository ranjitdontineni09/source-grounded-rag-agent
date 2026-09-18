# Connect-style stream

Near-real-time telemetry pattern: events on a queue, five-minute microbatches, partitioned tables instead of a 24-hour batch.

Dedup uses (contact id, event, timestamp). Watermarks drop late events. Hot-vs-cold validation compares fresh partitions to the historical daily file.

Production mapping: SQS + Lambda, Kinesis, Spark on EMR Serverless, Iceberg.
