import time
import psutil
import threading
from dataclasses import dataclass
from typing import Optional

try:
    import pynvml
    pynvml.nvmlInit()
    GPU_AVAILABLE = True
except Exception:
    GPU_AVAILABLE = False

from prometheus_client import Gauge, Histogram, Counter, start_http_server

GPU_MEM_USED  = Gauge("gpu_memory_used_mb",  "GPU memory used (MB)")
GPU_MEM_TOTAL = Gauge("gpu_memory_total_mb", "GPU memory total (MB)")
GPU_UTIL      = Gauge("gpu_utilization_pct", "GPU utilization %")
CPU_UTIL      = Gauge("cpu_utilization_pct", "CPU utilization %")
RAM_USED      = Gauge("ram_used_mb",         "RAM used (MB)")
RAM_TOTAL     = Gauge("ram_total_mb",        "RAM total (MB)")
INFERENCE_LAT = Histogram("inference_latency_seconds", "Per-request latency",
                           buckets=[0.01, 0.05, 0.1, 0.25, 0.5, 1.0, 2.0, 5.0])
REQUEST_COUNT = Counter("inference_requests_total", "Total inference requests")
ERROR_COUNT   = Counter("inference_errors_total",   "Total inference errors")


@dataclass
class InfraSnapshot:
    timestamp:       float
    cpu_pct:         float
    ram_used_mb:     float
    ram_total_mb:    float
    gpu_mem_used_mb:  Optional[float]
    gpu_mem_total_mb: Optional[float]
    gpu_util_pct:     Optional[float]


class InfraMonitor:
    def __init__(self, gpu_index: int = 0, poll_interval: float = 10.0, prometheus_port: int = 8001):
        self.gpu_index     = gpu_index
        self.poll_interval = poll_interval
        self._snapshots: list[InfraSnapshot] = []
        self._running = False
        start_http_server(prometheus_port)

    def snapshot(self) -> InfraSnapshot:
        cpu = psutil.cpu_percent(interval=0.1)
        ram = psutil.virtual_memory()
        gpu_used = gpu_total = gpu_util = None
        if GPU_AVAILABLE:
            handle    = pynvml.nvmlDeviceGetHandleByIndex(self.gpu_index)
            mem       = pynvml.nvmlDeviceGetMemoryInfo(handle)
            util      = pynvml.nvmlDeviceGetUtilizationRates(handle)
            gpu_used  = mem.used  / 1024**2
            gpu_total = mem.total / 1024**2
            gpu_util  = util.gpu
        snap = InfraSnapshot(
            timestamp=time.time(),
            cpu_pct=cpu,
            ram_used_mb=ram.used   / 1024**2,
            ram_total_mb=ram.total / 1024**2,
            gpu_mem_used_mb=gpu_used,
            gpu_mem_total_mb=gpu_total,
            gpu_util_pct=gpu_util,
        )
        CPU_UTIL.set(cpu)
        RAM_USED.set(snap.ram_used_mb)
        RAM_TOTAL.set(snap.ram_total_mb)
        if gpu_used  is not None: GPU_MEM_USED.set(gpu_used)
        if gpu_total is not None: GPU_MEM_TOTAL.set(gpu_total)
        if gpu_util  is not None: GPU_UTIL.set(gpu_util)
        self._snapshots.append(snap)
        if len(self._snapshots) > 1000:
            self._snapshots.pop(0)
        return snap

    def start(self):
        self._running = True
        threading.Thread(target=self._poll_loop, daemon=True).start()

    def stop(self):
        self._running = False

    def _poll_loop(self):
        while self._running:
            self.snapshot()
            time.sleep(self.poll_interval)

    def record_request(self, success: bool = True):
        REQUEST_COUNT.inc()
        if not success:
            ERROR_COUNT.inc()

    def latency_timer(self):
        return INFERENCE_LAT.time()

    def latest(self) -> Optional[InfraSnapshot]:
        return self._snapshots[-1] if self._snapshots else None
