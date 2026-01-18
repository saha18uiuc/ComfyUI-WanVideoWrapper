import os
import collections
import torch


class SchedulerGraphRunner:
    """
    Wraps a scheduler.step call inside a CUDA graph for repeated execution with static shapes.
    """

    def __init__(self, scheduler, scheduler_step_args, warmup_iters=2):
        self.scheduler = scheduler
        self.scheduler_step_args = scheduler_step_args or {}
        self.warmup_iters = warmup_iters
        self.graph = None
        self.static_noise = None
        self.static_latent = None
        self.static_timestep = None
        self.output = None
        self.enabled = torch.cuda.is_available()

    def _clone_static(self, tensor):
        return tensor.detach().clone()

    def release(self):
        self.graph = None
        self.static_noise = None
        self.static_latent = None
        self.static_timestep = None
        self.output = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def _capture(self, noise, timestep, latent):
        if not self.enabled or not noise.is_cuda:
            return
        self.static_noise = self._clone_static(noise)
        self.static_latent = self._clone_static(latent)
        self.static_timestep = self._clone_static(timestep)
        torch.cuda.synchronize()
        try:
            for _ in range(self.warmup_iters):
                self.output = self.scheduler.step(
                    self.static_noise,
                    self.static_timestep,
                    self.static_latent,
                    **self.scheduler_step_args
                )[0]
            torch.cuda.synchronize()
            self.graph = torch.cuda.CUDAGraph()
            with torch.cuda.graph(self.graph):
                self.output = self.scheduler.step(
                    self.static_noise,
                    self.static_timestep,
                    self.static_latent,
                    **self.scheduler_step_args
                )[0]
            torch.cuda.synchronize()
        except torch.cuda.OutOfMemoryError:
            self.release()
            raise

    def __call__(self, noise, timestep, latent):
        if not self.enabled or not noise.is_cuda:
            return self.scheduler.step(noise, timestep, latent, **self.scheduler_step_args)[0]
        if self.graph is None:
            self._capture(noise, timestep, latent)
        else:
            if noise.shape != self.static_noise.shape or latent.shape != self.static_latent.shape or timestep.shape != self.static_timestep.shape:
                raise ValueError("SchedulerGraphRunner input shape changed, cannot reuse captured graph.")
            self.static_noise.copy_(noise)
            self.static_latent.copy_(latent)
            self.static_timestep.copy_(timestep)
            try:
                self.graph.replay()
            except torch.cuda.OutOfMemoryError:
                self.release()
                raise
        return self.output


class GraphDenoiser:
    """
    Captures a transformer forward pass into a CUDA graph with static tensor buffers.
    Requires inputs to keep identical tree structure between invocations.
    """

    def __init__(
        self,
        model,
        warmup_iters=2,
        memory_factor=2.0,
        min_free_ratio=0.3,
        max_capture_bytes=None,
        device=None,
    ):
        self.model = model
        self.warmup_iters = warmup_iters
        self.memory_factor = memory_factor
        self.min_free_ratio = min_free_ratio
        self.max_capture_bytes = max_capture_bytes
        self.device = device
        self.enabled = torch.cuda.is_available()
        self.capture_stream = None
        if (
            self.enabled
            and isinstance(self.device, torch.device)
            and self.device.type == "cuda"
        ):
            self.capture_stream = torch.cuda.Stream(device=self.device)
        self.graph_cache = collections.OrderedDict()
        self.max_cached_variants = max(
            1,
            int(os.environ.get("WAN_MAX_GRAPH_VARIANTS", "4"))
        )

    def _clone_structure(self, obj):
        if torch.is_tensor(obj):
            return obj.detach().clone()
        if isinstance(obj, list):
            return [self._clone_structure(o) for o in obj]
        if isinstance(obj, tuple):
            return tuple(self._clone_structure(o) for o in obj)
        if isinstance(obj, dict):
            return {k: self._clone_structure(v) for k, v in obj.items()}
        return obj

    def _copy_structure(self, static_obj, new_obj):
        if torch.is_tensor(static_obj):
            if not torch.is_tensor(new_obj):
                raise ValueError("Expected tensor input for CUDA graph.")
            if static_obj.shape != new_obj.shape or static_obj.dtype != new_obj.dtype:
                raise ValueError("CUDA graph tensor shape mismatch.")
            static_obj.copy_(new_obj)
            return
        if isinstance(static_obj, list):
            if not isinstance(new_obj, list) or len(static_obj) != len(new_obj):
                raise ValueError("CUDA graph list structure changed.")
            for s_item, n_item in zip(static_obj, new_obj):
                self._copy_structure(s_item, n_item)
            return
        if isinstance(static_obj, tuple):
            if not isinstance(new_obj, tuple) or len(static_obj) != len(new_obj):
                raise ValueError("CUDA graph tuple structure changed.")
            for s_item, n_item in zip(static_obj, new_obj):
                self._copy_structure(s_item, n_item)
            return
        if isinstance(static_obj, dict):
            if not isinstance(new_obj, dict) or static_obj.keys() != new_obj.keys():
                raise ValueError("CUDA graph dict keys changed.")
            for key in static_obj.keys():
                self._copy_structure(static_obj[key], new_obj[key])
            return
        if static_obj != new_obj:
            raise ValueError("Non-tensor argument changed between CUDA graph replays.")

    def _estimate_bytes(self, obj):
        if torch.is_tensor(obj):
            return obj.nelement() * obj.element_size()
        if isinstance(obj, (list, tuple)):
            return sum(self._estimate_bytes(o) for o in obj)
        if isinstance(obj, dict):
            return sum(self._estimate_bytes(v) for v in obj.values())
        return 0

    def _clear_graph(self):
        self.graph_cache.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    def release(self):
        self._clear_graph()

    def _signature(self, obj):
        sig = []

        def encode(item):
            if torch.is_tensor(item):
                sig.append(
                    (
                        "tensor",
                        tuple(item.shape),
                        str(item.dtype),
                        item.device.type,
                    )
                )
                return
            if isinstance(item, (list, tuple)):
                sig.append(("list", len(item)))
                for sub in item:
                    encode(sub)
                return
            if isinstance(item, dict):
                sig.append(("dict", tuple(sorted(item.keys()))))
                for key in sorted(item.keys()):
                    encode(item[key])
                return
            sig.append(("obj", type(item).__name__))

        encode(obj)
        return tuple(sig)

    def _args_signature(self, args, kwargs):
        return (self._signature(args), self._signature(kwargs))

    def _capture(self, args, kwargs):
        if not self.enabled:
            raise RuntimeError("CUDA graphs not available on this device.")
        free_bytes, total_bytes = torch.cuda.mem_get_info()
        free_ratio = free_bytes / max(total_bytes, 1)
        required = self._estimate_bytes(args) + self._estimate_bytes(kwargs)
        if self.max_capture_bytes is not None and required > self.max_capture_bytes:
            raise RuntimeError("graph_insufficient_memory")
        if required == 0 or free_bytes < required * self.memory_factor or free_ratio < self.min_free_ratio:
            raise RuntimeError("graph_insufficient_memory")
        static_args = self._clone_structure(args)
        static_kwargs = self._clone_structure(kwargs)
        torch.cuda.synchronize(device=self.device)
        try:
            stream = self.capture_stream or torch.cuda.current_stream(device=self.device)
            current_stream = torch.cuda.current_stream(device=self.device)
            if stream is not current_stream:
                stream.wait_stream(current_stream)
            with torch.cuda.stream(stream):
                for _ in range(self.warmup_iters):
                    output = self.model(*static_args, **static_kwargs)
                stream.synchronize()
                graph = torch.cuda.CUDAGraph()
                with torch.cuda.graph(graph, stream=stream):
                    output = self.model(*static_args, **static_kwargs)
                stream.synchronize()
            if stream is not current_stream:
                current_stream.wait_stream(stream)
            return {
                "graph": graph,
                "static_args": static_args,
                "static_kwargs": static_kwargs,
                "output": output,
            }
        except torch.cuda.OutOfMemoryError:
            raise
        except RuntimeError as exc:
            raise RuntimeError(f"graph_capture_failed: {exc}") from exc

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.model(*args, **kwargs)
        signature = self._args_signature(args, kwargs)
        entry = self.graph_cache.get(signature)
        if entry is None:
            try:
                entry = self._capture(args, kwargs)
            except RuntimeError as exc:
                raise exc
            if len(self.graph_cache) >= self.max_cached_variants:
                self.graph_cache.popitem(last=False)
            self.graph_cache[signature] = entry
        else:
            self._copy_structure(entry["static_args"], args)
            self._copy_structure(entry["static_kwargs"], kwargs)
            try:
                stream = self.capture_stream or torch.cuda.current_stream(device=self.device)
                current_stream = torch.cuda.current_stream(device=self.device)
                if stream is not current_stream:
                    stream.wait_stream(current_stream)
                entry["graph"].replay()
                if stream is not current_stream:
                    current_stream.wait_stream(stream)
            except torch.cuda.OutOfMemoryError:
                self.graph_cache.pop(signature, None)
                raise
            except RuntimeError as exc:
                raise RuntimeError(f"graph_replay_failed: {exc}") from exc
        self.graph_cache.move_to_end(signature)
        return self.graph_cache[signature]["output"]
