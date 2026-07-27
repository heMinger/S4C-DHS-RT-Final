from torch import optim
from torch.optim.lr_scheduler import StepLR, LambdaLR, SequentialLR


class FixLR(optim.lr_scheduler._LRScheduler):
    def __init__(self, optimizer, last_epoch=-1, verbose=False):
        super(FixLR, self).__init__(optimizer, last_epoch, verbose)

    def get_lr(self):
        return [base_lr for base_lr in self.base_lrs]

    def _get_closed_form_lr(self):
        return [base_lr for base_lr in self.base_lrs]


def make_scheduler(config, optim):
    sched_type = config.get("type", "fix")
    warmup_steps = config.get("warmup_steps", 0)

    if sched_type == "fix":
        main_scheduler = FixLR(optim)
    elif sched_type == "step":
        main_scheduler = StepLR(
            optim,
            config["step_size"],
            config["gamma"]
        )
    else:
        raise NotImplementedError(f"Unknown learning rate scheduler type: {sched_type}")

    if warmup_steps > 0:
        warmup_scheduler = LambdaLR(
            optim,
            lr_lambda=lambda step: min(1.0, (step + 1) / warmup_steps)
        )
        scheduler = SequentialLR(
            optim,
            schedulers=[warmup_scheduler, main_scheduler],
            milestones=[warmup_steps]
        )
        return scheduler

    return main_scheduler
