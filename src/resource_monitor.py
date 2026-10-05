import os
import time
import threading
import psutil

def _as_int(value):
    if value is None:
        return None
    if hasattr(value, "value"):
        value = value.value
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _usable_vram_bytes(value):
    value = _as_int(value)
    # NVML usa 0xFFFFFFFFFFFFFFFF cuando el driver no reporta memoria del proceso.
    if value is None or value < 0 or value >= (1 << 62):
        return None
    return value


def _parse_size_to_bytes(value):
    parts = value.split()
    if not parts:
        return None
    try:
        amount = float(parts[0])
    except ValueError:
        return None
    unit = parts[1].lower() if len(parts) > 1 else "b"
    scale = {
        "b": 1,
        "byte": 1,
        "bytes": 1,
        "kib": 1024,
        "kb": 1024,
        "mib": 1024 ** 2,
        "mb": 1024 ** 2,
    }
    if unit not in scale:
        return None
    return int(amount * scale[unit])


def _parse_ns(value):
    parts = value.split()
    if not parts:
        return None
    try:
        return int(float(parts[0]))
    except ValueError:
        return None


def _parse_amd_fdinfo(text):
    """Un fd de /proc/<pid>/fdinfo. None si no es un cliente amdgpu."""
    client_id = None
    pdev = ""
    vram = 0
    engines = 0
    is_amd = False
    for line in text.splitlines():
        if ":" not in line:
            continue
        key, value = line.split(":", 1)
        key = key.strip()
        value = value.strip()
        if key == "drm-driver":
            is_amd = value == "amdgpu"
        elif key == "drm-client-id":
            client_id = value
        elif key == "drm-pdev":
            pdev = value
        elif key == "drm-memory-vram":
            parsed = _parse_size_to_bytes(value)
            if parsed is not None:
                vram = parsed
        elif key in ("drm-engine-gfx", "drm-engine-compute"):
            parsed = _parse_ns(value)
            if parsed is not None:
                engines += parsed
    if not is_amd:
        return None
    return client_id, pdev, vram, engines


def _aggregate_device_usage(device_stats):
    """Suma la VRAM del proceso y promedia la utilización de las GPUs donde está.

    device_stats: lista de {"mem_bytes": int | None, "sm_util": float | None}.
    sm_util None significa que la ventana de utilización todavía no es válida.
    """
    total_mem = 0
    mem_found = False
    utils = []
    for dev in device_stats:
        mem = dev.get("mem_bytes")
        sm = dev.get("sm_util")
        if mem is not None:
            total_mem += mem
            mem_found = True
        if sm is not None and (mem is not None or sm > 0):
            utils.append(min(float(sm), 100.0))
    util = sum(utils) / len(utils) if utils else None
    return util, total_mem if mem_found else None


def _amdgpu_present():
    drm = "/sys/class/drm"
    try:
        names = os.listdir(drm)
    except OSError:
        return False
    for name in names:
        if not (name.startswith("card") and name[4:].isdigit()):
            continue
        vendor_path = os.path.join(drm, name, "device", "vendor")
        try:
            with open(vendor_path) as handle:
                vendor = handle.read().strip().lower()
        except OSError:
            continue
        if vendor == "0x1002":
            return True
    return False


class GPUMonitor:
    def __init__(self):
        self.backend = None
        self.pynvml = None
        self.nvml_handles = []
        self._util_ts = []
        self._util_primed = []
        self._amd_prev = {}
        self._amd_prev_ns = None

        try:
            import pynvml
            pynvml.nvmlInit()
            count = pynvml.nvmlDeviceGetCount()
            if count < 1:
                raise RuntimeError("no hay GPUs NVIDIA")
            self.nvml_handles = [pynvml.nvmlDeviceGetHandleByIndex(i) for i in range(count)]
            self._util_ts = [0] * count
            self._util_primed = [False] * count
            self.pynvml = pynvml
            self.backend = "nvidia"
        except Exception:
            print("No se pudo inicializar el monitor de GPU NVIDIA")

        if self.backend is None and _amdgpu_present():
            self.backend = "amd"

        if self.backend is None:
            print("No se pudo inicializar el monitor de GPU")

    def is_available(self):
        return self.backend is not None

    def get_gpu_usage(self, pids):
        """Utilización (%) y VRAM (MB) solo de los PID indicados."""
        try:
            pid_set = {int(pid) for pid in pids}
            if not pid_set:
                return None, None
            if self.backend == "nvidia":
                return self._nvidia_usage(pid_set)
            if self.backend == "amd":
                return self._amd_usage(pid_set)
        except Exception:
            return None, None
        return None, None

    def _nvml_process_list(self, getter_name, handle):
        getter = getattr(self.pynvml, getter_name, None)
        if getter is None:
            return []
        try:
            return list(getter(handle) or [])
        except Exception:
            return []

    def _nvidia_memory_by_pid(self, handle, pids):
        mem_by_pid = {}
        for getter_name in (
            "nvmlDeviceGetComputeRunningProcesses",
            "nvmlDeviceGetGraphicsRunningProcesses",
        ):
            for proc in self._nvml_process_list(getter_name, handle):
                pid = _as_int(getattr(proc, "pid", None))
                if pid not in pids:
                    continue
                mem = _usable_vram_bytes(getattr(proc, "usedGpuMemory", None))
                if mem is None:
                    continue
                mem_by_pid[pid] = max(mem_by_pid.get(pid, 0), mem)
        return mem_by_pid

    def _nvidia_sm_util(self, handle, index, pids):
        fn = getattr(self.pynvml, "nvmlDeviceGetProcessUtilization", None)
        if fn is None:
            return None
        try:
            samples = list(fn(handle, self._util_ts[index]) or [])
        except Exception:
            return None

        newest = self._util_ts[index]
        sm = 0.0
        for sample in samples:
            timestamp = _as_int(getattr(sample, "timeStamp", None)) or 0
            if timestamp > newest:
                newest = timestamp
            pid = _as_int(getattr(sample, "pid", None))
            if pid in pids:
                sm += float(_as_int(getattr(sample, "smUtil", None)) or 0)

        if not samples:
            newest = max(newest, int(time.time() * 1_000_000))
        self._util_ts[index] = newest

        # La primera lectura con timestamp 0 cubre desde que cargó el driver.
        if not self._util_primed[index]:
            self._util_primed[index] = True
            return None
        return min(sm, 100.0)

    def _nvidia_usage(self, pids):
        device_stats = []
        for index, handle in enumerate(self.nvml_handles):
            mem_by_pid = self._nvidia_memory_by_pid(handle, pids)
            mem_bytes = sum(mem_by_pid.values()) if mem_by_pid else None
            device_stats.append({
                "mem_bytes": mem_bytes,
                "sm_util": self._nvidia_sm_util(handle, index, pids),
            })
        util, mem_bytes = _aggregate_device_usage(device_stats)
        mem_mb = None if mem_bytes is None else mem_bytes / (1024 ** 2)
        return util, mem_mb

    def _read_pid_amd(self, pid):
        """VRAM y tiempo de motores por dispositivo, sin contar dos veces el mismo cliente."""
        fd_dir = f"/proc/{pid}/fdinfo"
        try:
            fds = os.listdir(fd_dir)
        except OSError:
            return None

        clients = {}
        for fd in fds:
            path = os.path.join(fd_dir, fd)
            try:
                with open(path, encoding="utf-8", errors="ignore") as handle:
                    parsed = _parse_amd_fdinfo(handle.read())
            except OSError:
                continue
            if parsed is None:
                continue
            client_id, pdev, vram, engines = parsed
            key = (pdev, client_id if client_id is not None else f"fd-{fd}")
            clients[key] = (vram, engines)

        if not clients:
            return None

        per_device = {}
        for (pdev, _), (vram, engines) in clients.items():
            prev_vram, prev_engines = per_device.get(pdev, (0, 0))
            per_device[pdev] = (prev_vram + vram, prev_engines + engines)
        return per_device

    def _amd_usage(self, pids):
        now = time.monotonic_ns()
        per_pdev_vram = {}
        current = {}

        for pid in pids:
            devices = self._read_pid_amd(pid)
            if not devices:
                continue
            for pdev, (vram, engines) in devices.items():
                per_pdev_vram[pdev] = per_pdev_vram.get(pdev, 0) + vram
                current[(pid, pdev)] = current.get((pid, pdev), 0) + engines

        util = None
        if self._amd_prev_ns is not None and current:
            elapsed = now - self._amd_prev_ns
            delta_by_pdev = {}
            for key, engines in current.items():
                prev = self._amd_prev.get(key)
                if prev is None:
                    continue
                pdev = key[1]
                delta_by_pdev[pdev] = delta_by_pdev.get(pdev, 0) + max(0, engines - prev)
            if elapsed > 0 and delta_by_pdev:
                utils = [
                    min(100.0, delta / elapsed * 100.0)
                    for delta in delta_by_pdev.values()
                ]
                util = sum(utils) / len(utils)

        if current:
            self._amd_prev = current
            self._amd_prev_ns = now

        mem_mb = None
        if per_pdev_vram:
            mem_mb = sum(per_pdev_vram.values()) / (1024 ** 2)
        return util, mem_mb


def get_current_pid():
    return os.getpid()


class ResourceMonitor:
    def __init__(self, pid, interval=0.001):
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
                pids = {p.pid for p in procs}
                gpu_util, gpu_mem = self._gpu_monitor.get_gpu_usage(pids)
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