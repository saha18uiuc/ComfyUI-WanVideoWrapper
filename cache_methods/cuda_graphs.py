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

    def _capture(self, noise, timestep, latent):
        if not self.enabled or not noise.is_cuda:
            return
        self.static_noise = self._clone_static(noise)
        self.static_latent = self._clone_static(latent)
        self.static_timestep = self._clone_static(timestep)
        torch.cuda.synchronize()
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
            self.graph.replay()
        return self.output


class GraphDenoiser:
    """
    Captures a transformer forward pass into a CUDA graph with static tensor buffers.
    Requires inputs to keep identical tree structure between invocations.
    """

    def __init__(self, model, warmup_iters=2):
        self.model = model
        self.warmup_iters = warmup_iters
        self.graph = None
        self.static_args = None
        self.static_kwargs = None
        self.output = None
        self.enabled = torch.cuda.is_available()

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

    def _capture(self, args, kwargs):
        if not self.enabled:
            raise RuntimeError("CUDA graphs not available on this device.")
        self.static_args = self._clone_structure(args)
        self.static_kwargs = self._clone_structure(kwargs)
        torch.cuda.synchronize()
        for _ in range(self.warmup_iters):
            self.output = self.model(*self.static_args, **self.static_kwargs)
        torch.cuda.synchronize()
        self.graph = torch.cuda.CUDAGraph()
        with torch.cuda.graph(self.graph):
            self.output = self.model(*self.static_args, **self.static_kwargs)
        torch.cuda.synchronize()

    def __call__(self, *args, **kwargs):
        if not self.enabled:
            return self.model(*args, **kwargs)
        if self.graph is None:
            self._capture(args, kwargs)
        else:
            self._copy_structure(self.static_args, args)
            self._copy_structure(self.static_kwargs, kwargs)
            self.graph.replay()
        return self.output
