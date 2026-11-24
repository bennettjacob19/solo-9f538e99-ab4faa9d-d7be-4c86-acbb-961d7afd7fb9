"""读带 / 写盘流水线专项测试。

覆盖：

* 块 0 长写盘延迟、暂存仅 1 块时，块 1 在块 0 写完前已读完并通过 SHA-256；
* 暂存满且写盘者忙时不报错，写盘一空出、下一个应写块直接交接；
* 暂存满且写盘者空闲时仍按原规则报暂存额度不足；
* 读带侧坏块失败后，写盘侧只做完手头那一块；写盘侧失败后，读带侧
  只做完已开始的读帧，且 simulate_recovery 等两侧停稳才返回；
* 同样的带卷和额度，优化前后实读块数完全一致。
"""

import os
import sys
import tempfile
import threading
import time
import unittest

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from app.engine import (  # noqa: E402
    MemorySink,
    parse_write_delays,
    simulate_recovery,
)
from app.storage import load_set, sha256_hex  # noqa: E402

from helpers import ORDERED_LAYOUT, SCRAMBLED_LAYOUT, build_set, make_block  # noqa: E402


class ControlledSleep:
    """按线程区分读带/写盘等待；指定块的写盘等待可被测试主线程放行。

    引擎对块 ``blocked_block`` 的写盘延迟调用会被拦截并挂起，直到测试
    主线程置位 release_write；其余等待照常 sleep。
    """

    def __init__(self, read_delay: float, blocked_block: int | None = None) -> None:
        self.read_delay = read_delay
        self.blocked_block = blocked_block
        self.write_sleep_started = threading.Event()
        self.release_write = threading.Event()
        self._writer_sleep_index = 0

    def __call__(self, delay: float) -> None:
        in_writer = threading.current_thread().name == "recovery-writer"
        if in_writer:
            block_index = self._writer_sleep_index
            self._writer_sleep_index += 1
            if block_index == self.blocked_block:
                self.write_sleep_started.set()
                assert self.release_write.wait(timeout=10), "写盘等待未被放行，疑似死锁"
                return
        time.sleep(delay)


class BlockRecorderSink(MemorySink):
    """记录每块写盘顺序，便于断言读写重叠与顺序追加。"""

    def __init__(self) -> None:
        super().__init__()
        self.lock = threading.Lock()
        self.write_order: list[int] = []

    def write_block(self, index: int, data: bytes) -> None:
        super().write_block(index, data)
        with self.lock:
            self.write_order.append(index)


class FailingSink(MemorySink):
    """写到指定块号时抛出异常，模拟写盘侧失败。"""

    def __init__(self, fail_at: int, message: str) -> None:
        super().__init__()
        self.fail_at = fail_at
        self.message = message

    def write_block(self, index: int, data: bytes) -> None:
        if index == self.fail_at:
            raise RuntimeError(self.message)
        super().write_block(index, data)


class PipelineTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.mkdtemp()
        build_set(self.tmp, "ordered", ORDERED_LAYOUT)
        build_set(self.tmp, "scrambled", SCRAMBLED_LAYOUT)
        build_set(self.tmp, "bad", SCRAMBLED_LAYOUT, corrupt=11)

    def test_block1_read_and_verified_while_block0_writing(self):
        """题面场景：健康顺序带、暂存 1 块、块 0 写盘足够慢。"""

        tape_set = load_set(self.tmp, "ordered")
        sink = BlockRecorderSink()
        sleeper = ControlledSleep(read_delay=0.02, blocked_block=0)

        def sleep(delay: float) -> None:
            sleeper(delay)

        snapshots: list[tuple[int, int, int]] = []
        snapshots_lock = threading.Lock()

        def on_progress(metrics) -> None:
            # 在引擎持锁内回调：快照与状态变化同步，不会错过中间状态。
            with snapshots_lock:
                snapshots.append(
                    (
                        metrics.blocks_read,
                        metrics.staging_peak,
                        metrics.overlapped_reads,
                    )
                )

        thread_result = {}
        worker = threading.Thread(
            target=lambda: thread_result.update(
                result=simulate_recovery(
                    tape_set,
                    12,
                    1,
                    sink=sink,
                    read_delay=0.02,
                    write_delays={0: 100.0},  # 实际被 ControlledSleep 拦截
                    sleep=sleep,
                    progress=on_progress,
                )
            )
        )
        worker.start()
        self.assertTrue(sleeper.write_sleep_started.wait(5), "未观察到块 0 写盘等待")

        # 块 0 尚未写完：等到指标快照里出现“已读 ≥2 帧”，即块 1 已读完
        # 并通过入暂存前 SHA-256；此时写盘侧一块都不许写完。
        deadline = time.time() + 5
        while time.time() < deadline:
            with snapshots_lock:
                read_at_least_two = any(s[0] >= 2 for s in snapshots)
            if read_at_least_two:
                break
            if "result" in thread_result:
                self.fail("块 0 写盘未放行任务就结束了")
            time.sleep(0.02)
        with snapshots_lock:
            self.assertTrue(
                any(s[0] >= 2 for s in snapshots),
                "块 0 写完前未观察到块 1 读带并通过校验",
            )
        with sink.lock:
            self.assertEqual(
                sink.write_order,
                [],
                "块 1 校验完成时块 0 已写完，重叠未成立",
            )

        sleeper.release_write.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive(), "两侧线程未停稳")

        result = thread_result["result"]
        self.assertTrue(result.success, result.error)
        # 块 1 必然与块 0 的写盘发生过重叠。
        self.assertGreaterEqual(result.metrics.overlapped_reads, 1)
        self.assertEqual(result.metrics.blocks_read, 12)
        self.assertEqual(result.metrics.staging_peak, 1)
        self.assertEqual(sink.write_order, list(range(12)))
        # 逐块内容与摘要仍然正确（入暂存前 + 写出前双校验）。
        for index, chunk in enumerate(sink.chunks):
            self.assertEqual(chunk, make_block(index))
            self.assertEqual(
                sha256_hex(chunk), tape_set.entry(index).sha256
            )

    def test_full_staging_direct_handoff_when_writer_frees(self):
        """暂存满 + 写盘者忙：等待；写盘一空出，下一个应写块直接交接。"""

        # 布局：位置 0..3 分别是块 0,2,1,3；暂存额度仅 1，块 0 写盘慢。
        layout = [(0, 0, 0), (2, 0, 1), (1, 0, 2), (3, 0, 3)]
        build_set(self.tmp, "hand-off", layout)
        tape_set = load_set(self.tmp, "hand-off")
        sink = BlockRecorderSink()
        sleeper = ControlledSleep(read_delay=0.01, blocked_block=0)

        def sleep(delay: float) -> None:
            sleeper(delay)

        holder = {}
        worker = threading.Thread(
            target=lambda: holder.update(
                result=simulate_recovery(
                    tape_set,
                    10,
                    1,
                    sink=sink,
                    read_delay=0.01,
                    write_delays={0: 100.0},
                    sleep=sleep,
                )
            )
        )
        worker.start()
        self.assertTrue(sleeper.write_sleep_started.wait(5))
        # 读到块 1（第 3 帧）后它会在 place() 里等写盘者空出；
        # 给读带侧足够时间推进到“暂存满、等待写盘”的卡点。
        time.sleep(0.2)
        self.assertTrue(worker.is_alive(), "等待写盘期间任务意外结束（可能误报额度不足）")
        sleeper.release_write.set()
        worker.join(timeout=10)
        self.assertFalse(worker.is_alive())

        result = holder["result"]
        self.assertTrue(result.success, result.error)
        self.assertEqual(result.metrics.blocks_read, 4)  # 实读量不因优化增加
        self.assertEqual(result.metrics.staging_peak, 1)
        self.assertEqual(sink.write_order, [0, 1, 2, 3])

    def test_full_staging_idle_writer_still_quota_error(self):
        """暂存满且写盘者空闲：按原规则立即报暂存额度不足。"""

        layout = [(1, 0, 0), (0, 0, 1)]
        build_set(self.tmp, "quota", layout)
        tape_set = load_set(self.tmp, "quota")
        result = simulate_recovery(tape_set, 10, 0)
        self.assertFalse(result.success)
        self.assertIn("暂存额度不足", result.error)

        # 给够 1 块暂存则照常成功，指标与旧实现一致。
        ok = simulate_recovery(tape_set, 10, 1)
        self.assertTrue(ok.success, ok.error)
        self.assertEqual(ok.metrics.blocks_read, 2)
        self.assertEqual(ok.metrics.staging_peak, 1)

    def test_reader_bad_block_writer_finishes_only_current_block(self):
        tape_set = load_set(self.tmp, "bad")
        sink = MemorySink()
        result = simulate_recovery(
            tape_set, 100, 12, sink=sink, read_delay=0.01, sleep=time.sleep
        )
        self.assertFalse(result.success)
        self.assertEqual(result.metrics.bad_block, 11)
        # 坏块是 tape1 位置 0：tape0 的 6 帧 + 坏帧本身 = 7 次实读。
        self.assertEqual(result.metrics.blocks_read, 7)
        # 写盘侧只能消费已交接的块，不可能写出坏块或越过已投递范围。
        self.assertLessEqual(result.blocks_written, 6)
        self.assertEqual(sink.chunks, [make_block(i) for i in range(len(sink.chunks))])

    def test_writer_failure_reader_finishes_only_started_frame(self):
        tape_set = load_set(self.tmp, "ordered")
        sink = FailingSink(fail_at=1, message="模拟磁盘掉线")
        result = simulate_recovery(
            tape_set,
            12,
            1,
            sink=sink,
            read_delay=0.05,
            sleep=time.sleep,
        )
        self.assertFalse(result.success)
        self.assertIn("模拟磁盘掉线", result.error)
        self.assertEqual(result.blocks_written, 1)
        # 写盘在块 1 处失败；读带侧最多再做完已经开始的下一帧（块 2）。
        self.assertGreaterEqual(result.metrics.blocks_read, 2)
        self.assertLessEqual(result.metrics.blocks_read, 3)

    def test_read_volume_unchanged_same_quotas(self):
        """同样带卷与额度：流水线版实读量与基线（贪心）完全一致。"""

        for name in ("ordered", "scrambled"):
            tape_set = load_set(self.tmp, name)
            result = simulate_recovery(tape_set, 100, 12, read_delay=0.0)
            self.assertTrue(result.success, result.error)
            self.assertEqual(result.metrics.blocks_read, 12)
            self.assertEqual(result.metrics.rewinds, 0)

    def test_no_overlap_without_write_delay(self):
        tape_set = load_set(self.tmp, "ordered")
        result = simulate_recovery(tape_set, 12, 1)
        self.assertTrue(result.success)
        self.assertEqual(result.metrics.overlapped_reads, 0)

    def test_parse_write_delays(self):
        self.assertEqual(parse_write_delays(None), {})
        self.assertEqual(parse_write_delays(""), {})
        self.assertEqual(parse_write_delays("0:1.5, 3:0.2"), {0: 1.5, 3: 0.2})
        for bad in ("0", "a:1", "0:x", "-1:1", "1:-2"):
            with self.assertRaises(ValueError):
                parse_write_delays(bad)

    def test_invalid_arguments(self):
        tape_set = load_set(self.tmp, "ordered")
        with self.assertRaises(ValueError):
            simulate_recovery(tape_set, 12, 1, write_delay=-0.1)
        with self.assertRaises(ValueError):
            simulate_recovery(tape_set, 0, 1)


if __name__ == "__main__":
    unittest.main(verbosity=2)
