from dataclasses import dataclass


@dataclass
class SlidingWindowConfig:
    seq_len: int = 32
    cond_steps: int = 4
    stride: int = 1
    val_total_len: int = 128
    val_window_stride: int | None = None

    def __post_init__(self) -> None:
        if not (1 <= self.cond_steps <= self.seq_len):
            raise ValueError(
                f"Need 1 <= cond-steps <= seq-len; got cond_steps={self.cond_steps}, seq_len={self.seq_len}"
            )
        if self.val_total_len < self.cond_steps:
            raise ValueError(
                f"val_total_len ({self.val_total_len}) must be >= cond_steps ({self.cond_steps})"
            )
        if self.val_window_stride is None:
            self.val_window_stride = self.seq_len - self.cond_steps

