import os
import logging
from opentelemetry import trace, metrics, _logs
from opentelemetry.sdk.resources import Resource
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor
from opentelemetry.sdk.metrics import MeterProvider
from opentelemetry.sdk.metrics.export import PeriodicExportingMetricReader
from opentelemetry.sdk._logs import LoggerProvider, LoggingHandler
from opentelemetry.sdk._logs.export import BatchLogRecordProcessor

from opentelemetry.exporter.otlp.proto.http.trace_exporter import OTLPSpanExporter
from opentelemetry.exporter.otlp.proto.http.metric_exporter import OTLPMetricExporter
from opentelemetry.exporter.otlp.proto.http._log_exporter import OTLPLogExporter

# Instrumentations
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.instrumentation.logging import LoggingInstrumentor
from opentelemetry.instrumentation.requests import RequestsInstrumentor

# Constants for Grafana OTLP
GRAFANA_ENDPOINT = "https://otlp-gateway-prod-ap-south-1.grafana.net/otlp"
GRAFANA_AUTH_TOKEN = "Basic MTUyNzk4NTpnbGNfZXlKdklqb2lNVFkzTVRFNE1DSXNJbTRpT2lKbWFYSnpkQ0lzSW1zaU9pSlJjak15U1ZGTE5UUk5NRGhWT1VNeGIyWkVUelpOWXpBaUxDSnRJanA3SW5JaU9pSndjbTlrTFdGd0xYTnZkWFJvTFRFaWZYMD0="
GRAFANA_HEADERS = {"Authorization": GRAFANA_AUTH_TOKEN}

def init_telemetry(app=None):
    """
    Initializes OpenTelemetry Tracing, Metrics, and Logging.
    """
    resource = Resource.create({
        "service.name": "preprod-rag",
        "service.version": "1.0.0",
        "app.environment": "dev",
        "rag.component": "overall"
    })

    # 1. Tracing Setup
    trace_provider = TracerProvider(resource=resource)
    trace_exporter = OTLPSpanExporter(
        endpoint=f"{GRAFANA_ENDPOINT}/v1/traces",
        headers=GRAFANA_HEADERS
    )
    trace_provider.add_span_processor(BatchSpanProcessor(trace_exporter))
    trace.set_tracer_provider(trace_provider)

    # 2. Metrics Setup
    metric_exporter = OTLPMetricExporter(
        endpoint=f"{GRAFANA_ENDPOINT}/v1/metrics",
        headers=GRAFANA_HEADERS
    )
    metric_reader = PeriodicExportingMetricReader(metric_exporter)
    meter_provider = MeterProvider(resource=resource, metric_readers=[metric_reader])
    metrics.set_meter_provider(meter_provider)

    # 3. Logging Setup
    log_provider = LoggerProvider(resource=resource)
    log_exporter = OTLPLogExporter(
        endpoint=f"{GRAFANA_ENDPOINT}/v1/logs",
        headers=GRAFANA_HEADERS
    )
    log_provider.add_log_record_processor(BatchLogRecordProcessor(log_exporter))
    _logs.set_logger_provider(log_provider)

    # 4. Automatic Instrumentation
    LoggingInstrumentor().instrument(set_logging_format=True)
    RequestsInstrumentor().instrument()
    
    if app:
        FastAPIInstrumentor.instrument_app(app)

    return trace.get_tracer(__name__), metrics.get_meter(__name__)

# Singleton instances for convenience
tracer = trace.get_tracer("rag_tracer")
meter = metrics.get_meter("rag_meter")
