import os
import time
import threading
import psutil

class GPUMonitor:
    def __init__(self):
        self.backend = None
        self.pynvml = None 
        self.nvml_handle = None
        self.amd_gpu = None

        try:
            import pynvml
            pynvml.nvmlInit()
            self.nvml_handle = pynvml.nvmlDeviceGetHandleByIndex(0)
            self.pynvml = pynvml
            self.backend = "nvidia"
        except Exception:
            print("No se pudo inicializar el monitor de GPU NVIDIA")
        
        if self.backend is None:
            try:
                import pyamdgpuinfo
                if pyamdgpuinfo.detect_gpus() > 0:
                    self.amd_gpu = pyamdgpuinfo.get_gpu(0)
                    self.backend = "amd"
            except Exception:
                print("No se pudo inicializar el monitor de GPU AMD")
    
        if self.backend is None:
            print("No se pudo inicializar el monitor de GPU")

    def is_available(self):
        return self.backend is not None

    def get_gpu_usage(self):
        if self.backend == "nvidia":
            try:
                util = self.pynvml.nvmlDeviceGetUtilizationRates(self.nvml_handle)
                mem = self.pynvml.nvmlDeviceGetMemoryInfo(self.nvml_handle)
                return util.gpu, mem.used / (1024 ** 2) # Convert to MB
            except Exception:
                return None, None

        elif self.backend == "amd":
            try:
                util = self.amd_gpu.query_load() * 100 # Convert to percentage
                mem = self.amd_gpu.query_memory() / (1024 ** 2) # Convert to MB
                return util, mem
            except Exception:
                return None, None

        return None, None


def get_current_pid():
    return os.getpid()


class ResourceMonitor:
    def __init__(self, pid, interval=0.2):
        self.interval = interval
        self.root_pid = pid
        self.samples = {'cpu': [], 'memory': [], 'gpu_util': [], 'gpu_mem': []}
        self._stop_event = threading.Event()
        self._thread = None
        self._gpu_monitor = GPUMonitor()
    
    def _process_list(self):
        try:
            root = psutil.Process(self.root_pid)
            return [root] + root.children(recursive=True)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            return []

    def _run(self):
        procs = self._process_list()

        for p in procs:
            try:
                p.cpu_percent(interval=None)
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

        while not  self._stop_event.is_set():
            procs = self._process_list()
            cpu_total, ram_total = 0.0, 0.0
            for p in procs:
                try:
                    cpu_total += p.cpu_percent(interval=0.1)
                    ram_total += p.memory_info().rss / (1024 ** 2) # Convert to MB
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    pass
            
            self.samples['cpu'].append(cpu_total)
            self.samples['memory'].append(ram_total)

            if self.gpu_monitor_is_available():
                gpu_util, gpu_mem = self._gpu_monitor.get_gpu_usage()
                if gpu_util is not None:
                    self.samples['gpu_util'].append(gpu_util)
                if gpu_mem is not None:
                    self.samples['gpu_mem'].append(gpu_mem)

            time.sleep(self.interval)

    def start(self):
        self.samples = {'cpu': [], 'memory': [], 'gpu_util': [], 'gpu_mem': []}
        self._stop_event.clear()
        self._thread = threading.Thread(target=self._run, daemon=True)
        self._thread.start()

    def stop(self):
        self._stop_event.set()
        if self._thread:
            self._thread.join()

    def _calculate_stats(self, values):
        if not values:
            return None
        
        return { 
            "avg":sum(values) / len(values), 
            "min":min(values), 
            "max":max(values),
            "count":len(values)
        }
    
    def gpu_monitor_is_available(self):
        return self._gpu_monitor.is_available()

    def get_resource_usage(self):
        return {
            'cpu': self._calculate_stats(self.samples['cpu']),
            'memory': self._calculate_stats(self.samples['memory']),
            'gpu_util': self._calculate_stats(self.samples['gpu_util']),
            'gpu_mem': self._calculate_stats(self.samples['gpu_mem'])
        }