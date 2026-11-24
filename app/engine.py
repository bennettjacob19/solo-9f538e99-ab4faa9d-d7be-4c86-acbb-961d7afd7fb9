"""恢复演练引擎（读带 / 写盘流水线版）。

磁带语义（按演练规则实现）：

* 每盘磁带只能从头顺读；换带或回卷之后，下一次读取都从第 0 帧重新开始。
* 顺读路径上跳过的帧同样计入读带量；同一帧重复读到也重复计数。
* 提前读到、暂时不能按块号写出的块进入暂存区（内存或磁盘等价），
  暂存块数的峰值受暂存额度约束。
* 镜像只允许按块号 0、1、2…… 追加，不能回读、不能改前面的块。
* 每个块在入暂存前（读带侧）与写出前（写盘侧）都要再核对一次 SHA-256；
  坏块或任一额度不足立即失败，失败时不产出新镜像（由上层保留上一次成品）。

流水线语义（本次优化）：

* 读带线程负责机械动作（挂带/换带/回卷/顺读）、入暂存前摘要核对与投递；
  写盘线程是唯一的写盘者，严格按块号顺序追加，并承担写出前摘要核对与
  每块可配置的模拟写盘延迟（``write_delays`` / ``write_delay``）。
* 两线程并发后，写盘等待期间读带可以继续，缩短网页恢复的读带等待；
  读带机械语义与额度计量完全不变，同一带卷同一额度下实读块数不增加。
* 所有“前读后、等着追加”的已校验块都计入暂存；正在写盘中的那一块
  不占暂存（直接交接槽）。暂存满时，若下一个应写块到来，写盘者一旦
  空出即可直接交接，不会被暂存额度误杀；若写盘者空闲且仍放不下，
  说明额度确实不足，按原规则报暂存额度不足。
* 任一侧失败后，另一侧只允许做完已经开始的读帧或写盘，随后双双停稳；
  ``simulate_recovery`` 会等两个线程都结束才返回，上层据此接下一单。
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Callable, Mapping, Protocol

from .storage import BLOCK_SIZE, BlockEntry, TapeSet, read_frame, sha256_hex


class RecoveryError(RuntimeError):
    """恢复失败：坏块或额度不足。"""


class QuotaExceeded(RecoveryError):
    pass


class BadBlock(RecoveryError):
    pass


class ImageSink(Protocol):
    """镜像输出端：只允许按块号顺序追加。"""

    def write_block(self, index: int, data: bytes) -> None: ...

    def close(self) -> None: ...


class FileImageSink:
    """写临时镜像文件；只有全部块写完并核对通过后才由上层原子发布。"""

    def __init__(self, tmp_path: str) -> None:
        self._tmp_path = tmp_path
        self._fh = open(tmp_path, "wb")  # noqa: SIM115 - 生命周期由 close 管理
        self._next_index = 0
        self.bytes_written = 0

    def write_block(self, index: int, data: bytes) -> None:
        if index != self._next_index:
            raise RecoveryError(
                f"镜像只允许顺序追加：期望块 {self._next_index}，收到块 {index}"
            )
        if len(data) != BLOCK_SIZE:
            raise RecoveryError(f"块 {index} 不是 {BLOCK_SIZE} 字节")
        self._fh.write(data)
        self._fh.flush()
        os.fsync(self._fh.fileno())
        self._next_index += 1
        self.bytes_written += len(data)

    def close(self) -> None:
        if not self._fh.closed:
            self._fh.close()

    @property
    def next_index(self) -> int:
        return self._next_index


class MemorySink:
    """测试用内存输出端。"""

    def __init__(self) -> None:
        self.chunks: list[bytes] = []

    def write_block(self, index: int, data: bytes) -> None:
        if index != len(self.chunks):
            raise RecoveryError(
                f"镜像只允许顺序追加：期望块 {len(self.chunks)}，收到块 {index}"
            )
        self.chunks.append(data)

    def close(self) -> None:
        return None


@dataclass
class RecoveryMetrics:
    blocks_read: int = 0           # 实际读带块数（含跳过、含重读）
    staging_peak: int = 0          # 暂存块数峰值（不含正在写盘的直接交接块）
    tape_mounts: int = 0           # 换带次数（含第一次挂带）
    tape_switches: int = 0         # 在两盘带之间切换的次数
    rewinds: int = 0               # 同一盘带回卷次数
    overlapped_reads: int = 0      # 读帧完成时写盘仍在进行的读帧数（读写重叠）
    bad_block: int | None = None   # 命中的坏块号（若有）
    head_position: int = 0         # 当前磁头：本盘带上“下一个待读帧”的序号
    events: list[str] = field(default_factory=list)

    def log(self, message: str) -> None:
        self.events.append(message)


@dataclass
class RecoveryResult:
    success: bool
    set_name: str
    metrics: RecoveryMetrics
    error: str | None = None
    artifact_path: str | None = None
    blocks_written: int = 0


def parse_write_delays(raw: str | None) -> dict[int, float]:
    """解析 ``块号:秒数`` 形式的每块写盘延迟配置，如 ``"0:1.5,3:0.2"``。"""

    if not raw:
        return {}
    delays: dict[int, float] = {}
    for part in raw.split(","):
        part = part.strip()
        if not part:
            continue
        if ":" not in part:
            raise ValueError(f"写盘延迟配置格式应为 块号:秒数：{part!r}")
        key, value = part.split(":", 1)
        try:
            index = int(key.strip())
            delay = float(value.strip())
        except ValueError as exc:
            raise ValueError(f"写盘延迟配置无法解析：{part!r}") from exc
        if index < 0 or delay < 0:
            raise ValueError(f"写盘延迟配置出现负数：{part!r}")
        delays[index] = delay
    return delays


def _baseline_metrics(tape_set: TapeSet) -> RecoveryMetrics:
    """贪心策略（提前读到的块一律暂存）下的确定性指标，供页面预估额度。"""

    metrics = RecoveryMetrics()
    current_tape: int | None = None
    head_position = 0
    staged: dict[int, bytes] = {}
    written = 0
    n = tape_set.block_count

    for target in range(n):
        if target in staged:
            staged.pop(target, None)
            written += 1
            continue
        entry = tape_set.entry(target)
        if current_tape is None:
            current_tape = entry.tape
            head_position = 0
            metrics.tape_mounts += 1
            metrics.log(f"挂带 tape{entry.tape}")
        elif entry.tape != current_tape:
            current_tape = entry.tape
            head_position = 0
            metrics.tape_mounts += 1
            metrics.tape_switches += 1
            metrics.log(f"换带 tape{entry.tape}，从头顺读")
        elif entry.position < head_position:
            metrics.rewinds += 1
            head_position = 0
            metrics.log(f"回卷 tape{entry.tape}，从头顺读")
        # 目标帧在磁头之后：磁头自然前移，无需回卷。
        metrics.blocks_read += entry.position + 1 - head_position
        for position in range(head_position, entry.position + 1):
            frame_entry = _entry_at(tape_set, entry.tape, position)
            if (
                frame_entry.index >= written
                and frame_entry.index != target
                and frame_entry.index not in staged
            ):
                staged[frame_entry.index] = b""
        head_position = entry.position + 1
        metrics.staging_peak = max(metrics.staging_peak, len(staged))
        staged.pop(target, None)
        written += 1
    return metrics


def _entry_at(tape_set: TapeSet, tape: int, position: int) -> BlockEntry:
    for entry in tape_set.blocks:
        if entry.tape == tape and entry.position == position:
            return entry
    raise RecoveryError(f"清单缺少 tape{tape} 位置 {position} 的记录")


def baseline(tape_set: TapeSet) -> dict[str, int]:
    """返回贪心恢复所需的最小读带量与暂存峰值（坏块不影响位置推算）。"""

    metrics = _baseline_metrics(tape_set)
    return {
        "blocks_read": metrics.blocks_read,
        "staging_peak": metrics.staging_peak,
        "tape_mounts": metrics.tape_mounts,
        "tape_switches": metrics.tape_switches,
        "rewinds": metrics.rewinds,
    }


class _Pipeline:
    """读带线程与写盘线程之间的协调器。

    ``staged`` 是有界暂存区（受暂存额度约束）；``direct`` 是唯一的直接
    交接槽，承载正在写盘中的块，不计入暂存。所有共享状态都在 ``cv`` 下
    访问，任何状态变化都通知两侧重新评估。
    """

    def __init__(
        self,
        tape_set: TapeSet,
        sink: ImageSink,
        metrics: RecoveryMetrics,
        read_quota: int,
        staging_quota: int,
        *,
        discard_future_blocks: bool,
        read_delay: float,
        write_delay: float,
        write_delays: Mapping[int, float],
        sleep: Callable[[float], None] | None,
        progress: Callable[[RecoveryMetrics], None] | None,
    ) -> None:
        self._tape_set = tape_set
        self._sink = sink
        self._metrics = metrics
        self._read_quota = read_quota
        self._staging_quota = staging_quota
        self._discard = discard_future_blocks
        self._read_delay = read_delay
        self._write_delay = write_delay
        self._write_delays = dict(write_delays)
        self._sleep = sleep
        self._progress = progress

        self.cv = threading.Condition()
        self.staged: dict[int, bytes] = {}
        self.direct: tuple[int, bytes] | None = None
        self.delivered: set[int] = set()   # 已投递（暂存/直接交接/已写）的块号
        self.next_index = 0                # 写盘者下一个应写块号
        self.writer_busy = False           # 写盘者是否正在写盘（含写盘延迟）
        self.closed = False                # 读带侧是否已结束
        self.error: Exception | None = None
        self.blocks_written = 0

    # ---- 共享状态小工具（均在 cv 内调用）----

    def _log(self, message: str) -> None:
        self._metrics.log(message)

    def _emit(self) -> None:
        if self._progress is not None:
            self._progress(self._metrics)

    # ---- 读带侧：把一个已校验的块投入流水线 ----

    def place(self, index: int, data: bytes) -> None:
        with self.cv:
            while True:
                if self.error is not None:
                    raise self.error
                if index in self.delivered:
                    return
                # 下一个应写块且写盘者空闲：直接交接，不占暂存。
                if (
                    index == self.next_index
                    and not self.writer_busy
                    and self.direct is None
                ):
                    self.staged.pop(index, None)
                    self.direct = (index, data)
                    self.delivered.add(index)
                    self._log(f"块 {index} 读带后直接交接写盘（不占暂存）")
                    self.cv.notify_all()
                    self._emit()
                    return
                # 暂存区有空位：进入暂存，等待按块号追加。
                if index not in self.staged and len(self.staged) < self._staging_quota:
                    self.staged[index] = data
                    self.delivered.add(index)
                    self._metrics.staging_peak = max(
                        self._metrics.staging_peak, len(self.staged)
                    )
                    self._log(
                        f"提前读到块 {index}，暂存（当前暂存 {len(self.staged)} 块，"
                        f"写盘进行中：{'是' if self.writer_busy else '否'}）"
                    )
                    self.cv.notify_all()
                    self._emit()
                    return
                # 放不下。写盘者正忙：等它做完当前写盘再评估（不算额度失败）。
                if self.writer_busy or self.direct is not None:
                    self.cv.wait()
                    continue
                # 写盘者空闲却仍放不下：暂存额度确实不足，按原规则立即失败。
                raise QuotaExceeded(
                    f"暂存额度不足：暂存峰值将超过上限 {self._staging_quota} 块"
                )

    # ---- 写盘线程主循环 ----

    def writer_main(self) -> None:
        try:
            n = self._tape_set.block_count
            while True:
                with self.cv:
                    item: tuple[int, bytes] | None = None
                    while self.error is None:
                        if self.next_index >= n:
                            return
                        if self.next_index in self.staged:
                            item = (
                                self.next_index,
                                self.staged.pop(self.next_index),
                            )
                            break
                        if (
                            self.direct is not None
                            and self.direct[0] == self.next_index
                        ):
                            item = self.direct
                            self.direct = None
                            break
                        if self.closed:
                            return
                        self.cv.wait()
                    if self.error is not None or item is None:
                        return
                    index, data = item
                    self.writer_busy = True

                # 写出前再次核对 SHA-256（暂存内容损坏也要拦住）。
                entry = self._tape_set.entry(index)
                if sha256_hex(data) != entry.sha256:
                    raise BadBlock(
                        f"块 {index} 写出前摘要核对失败（暂存内容损坏）"
                    )
                delay = self._write_delays.get(index, self._write_delay)
                if delay and self._sleep is not None:
                    self._sleep(delay)
                self._sink.write_block(index, data)
                with self.cv:
                    self.writer_busy = False
                    self.next_index += 1
                    self.blocks_written += 1
                    self._log(f"块 {index} 追加写入镜像成功")
                    self.cv.notify_all()
                    self._emit()
        except BaseException as exc:  # noqa: BLE001 - 写盘侧任何异常都通知读带侧
            with self.cv:
                if self.error is None:
                    self.error = exc
                    self._log(f"写盘失败：{exc}")
                self.writer_busy = False
                self.cv.notify_all()
                self._emit()

    # ---- 读带线程主循环（机械语义与单线程版完全一致）----

    def reader_main(self) -> None:
        tape_set = self._tape_set
        n = tape_set.block_count
        current_tape: int | None = None
        head_position = 0
        metrics = self._metrics
        try:
            while True:
                with self.cv:
                    if self.error is not None:
                        raise self.error
                    # 读带侧始终追赶“第一个尚未投递到流水线”的块；
                    # 已在暂存/直接交接/已写盘的块都算投递过，不重复读。
                    target = next(
                        (i for i in range(n) if i not in self.delivered), None
                    )
                    if target is None:
                        break

                entry = tape_set.entry(target)

                # 挂带 / 换带：磁头回到带头（位置 0）。
                if current_tape is None:
                    current_tape = entry.tape
                    head_position = 0
                    with self.cv:
                        metrics.tape_mounts += 1
                        self._log(f"挂带 tape{entry.tape}，从头顺读")
                elif entry.tape != current_tape:
                    current_tape = entry.tape
                    head_position = 0
                    with self.cv:
                        metrics.tape_mounts += 1
                        metrics.tape_switches += 1
                        self._log(f"换带 tape{entry.tape}，从头顺读")
                elif entry.position < head_position:
                    # 目标帧落在磁头之后已越过的位置：必须回卷再从头顺读。
                    with self.cv:
                        metrics.rewinds += 1
                        self._log(f"回卷 tape{current_tape}，从头顺读")
                    head_position = 0
                # 目标帧在磁头之后时磁头自然前移，跳过的帧同样计入读带量。
                read_from = head_position

                for position in range(read_from, entry.position + 1):
                    with self.cv:
                        if self.error is not None:
                            raise self.error
                        if metrics.blocks_read >= self._read_quota:
                            raise QuotaExceeded(
                                f"读带额度不足：已读 {metrics.blocks_read} 块，"
                                f"上限 {self._read_quota} 块"
                            )
                    data = read_frame(tape_set, current_tape, position)
                    if self._read_delay and self._sleep is not None:
                        self._sleep(self._read_delay)
                    frame_entry = _entry_at(tape_set, current_tape, position)

                    with self.cv:
                        if self.error is not None:
                            raise self.error
                        metrics.blocks_read += 1
                        metrics.head_position = position + 1
                        # 读帧完成时写盘仍在进行：该读帧与写盘重叠。
                        if self.writer_busy:
                            metrics.overlapped_reads += 1
                        head_position = position + 1

                    # 入暂存/交接前先核对摘要，坏块立即失败。
                    if sha256_hex(data) != frame_entry.sha256:
                        metrics.bad_block = frame_entry.index
                        raise BadBlock(
                            f"坏块：镜像块 {frame_entry.index}（tape{current_tape} "
                            f"位置 {position}）SHA-256 核对失败"
                        )

                    with self.cv:
                        already = frame_entry.index in self.delivered
                    if already:
                        # 顺读/重读路径上已投递或已写出的历史块：不重复暂存。
                        continue
                    if self._discard and frame_entry.index != target:
                        with self.cv:
                            self._log(
                                f"提前读到块 {frame_entry.index}，按策略丢弃，"
                                "稍后回卷重读"
                            )
                        continue
                    # 暂存满时这里会阻塞等待写盘推进；写盘者空闲且放不下
                    # 才抛暂存额度不足。
                    self.place(frame_entry.index, data)
        except BaseException as exc:  # noqa: BLE001 - 读带侧任何异常都通知写盘侧
            with self.cv:
                if self.error is None:
                    self.error = exc
                self.closed = True
                self.cv.notify_all()
                self._emit()
        else:
            with self.cv:
                self.closed = True
                self.cv.notify_all()
                self._emit()


def simulate_recovery(
    tape_set: TapeSet,
    read_quota: int,
    staging_quota: int,
    sink: ImageSink | None = None,
    *,
    discard_future_blocks: bool = False,
    read_delay: float = 0.0,
    sleep: Callable[[float], None] | None = None,
    write_delay: float = 0.0,
    write_delays: Mapping[int, float] | None = None,
    progress: Callable[[RecoveryMetrics], None] | None = None,
) -> RecoveryResult:
    """按给定额度执行恢复演练（读带 / 写盘流水线）。

    read_quota / staging_quota 为块数上限。sink 为 None 时只做可行性模拟
    （不写镜像），用于预估与单元测试。discard_future_blocks 用来模拟暂存
    资源受限、提前块被丢弃的策略，此时后续需要回卷重读。

    write_delay 是每块默认的模拟写盘等待秒数；write_delays 可给指定块号
    单独覆盖（如 ``{0: 1.5}``）。sleep 为注入的等待函数，测试可替换。
    progress 在指标变化时被回调（持锁内调用，回调内不要阻塞）。
    """

    if read_quota <= 0 or staging_quota < 0:
        raise ValueError("读带额度必须为正整数，暂存额度必须为非负整数")
    if write_delay < 0:
        raise ValueError("写盘延迟不能为负数")

    own_sink = sink is None
    if sink is None:
        sink = MemorySink()
    metrics = RecoveryMetrics()
    pipeline = _Pipeline(
        tape_set,
        sink,
        metrics,
        read_quota,
        staging_quota,
        discard_future_blocks=discard_future_blocks,
        read_delay=read_delay,
        write_delay=write_delay,
        write_delays=write_delays or {},
        sleep=sleep,
        progress=progress,
    )

    writer = threading.Thread(
        target=pipeline.writer_main, name="recovery-writer", daemon=True
    )
    reader = threading.Thread(
        target=pipeline.reader_main, name="recovery-reader", daemon=True
    )
    writer.start()
    reader.start()
    # 两侧都停稳后本函数才返回：上层据此保证“停稳后才接下一单”。
    reader.join()
    writer.join()

    written = pipeline.blocks_written
    error = pipeline.error
    try:
        if error is not None:
            if not any(e.startswith("恢复失败") for e in metrics.events):
                metrics.log(f"恢复失败：{error}")
            return RecoveryResult(
                success=False,
                set_name=tape_set.name,
                metrics=metrics,
                error=str(error),
                blocks_written=written,
            )
        if written != tape_set.block_count:
            raise RecoveryError(
                f"恢复不完整：应写 {tape_set.block_count} 块，实写 {written} 块"
            )
        metrics.log(
            f"恢复完成：共读带 {metrics.blocks_read} 块，暂存峰值 "
            f"{metrics.staging_peak} 块，与写盘重叠读帧 {metrics.overlapped_reads} 个"
        )
        return RecoveryResult(
            success=True,
            set_name=tape_set.name,
            metrics=metrics,
            blocks_written=written,
        )
    except RecoveryError as exc:
        metrics.log(f"恢复失败：{exc}")
        return RecoveryResult(
            success=False,
            set_name=tape_set.name,
            metrics=metrics,
            error=str(exc),
            blocks_written=written,
        )
    finally:
        if not own_sink:
            sink.close()
