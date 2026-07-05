#!/usr/bin/env python3
"""
PySide6 desktop GUI for gguf_vram.py (the llama.cpp VRAM estimator).

Unlike the tkinter version, Qt6 has a native Wayland backend and automatic
HiDPI scaling, so the UI is crisp and correctly sized out of the box — no
DPI/scaling code needed.

    pip install pyside6
    python gguf_vram_qt.py

Enter a Hugging Face repo id, optionally click "Fetch files" to list the
available .gguf files, set your token budget, and press "Estimate". Network
work runs in a background thread so the window never freezes.
"""

import os
import sys
import threading
import urllib.error
import urllib.parse

from PySide6.QtCore import Qt, Signal
from PySide6.QtGui import QFont, QFontDatabase
from PySide6.QtWidgets import (
    QApplication, QComboBox, QDoubleSpinBox, QGridLayout, QHBoxLayout, QLabel,
    QLineEdit, QPlainTextEdit, QPushButton, QSpinBox, QWidget,
)

import gguf_vram as core


def resolve_target(sizes, choice):
    """Given the fetched {path: size} map and the user's choice string,
    return (weight_bytes, meta_file, shown_name). Handles split GGUFs."""
    target = None
    if choice:
        if choice in sizes:                       # exact filename
            target = choice
        else:                                     # treat as quant substring
            matches = [p for p in sizes if choice.lower() in p.lower()]
            if not matches:
                raise ValueError(f"no file matches '{choice}'")
            target = sorted(matches, key=len)[0]
    elif len(sizes) == 1:
        target = next(iter(sizes))
    else:
        raise ValueError("multiple GGUF files: pick one or type a quant (e.g. Q4_K_M)")

    grp = core.split_group(target)
    if grp:
        parts = sorted(p for p in sizes if core.split_group(p) == grp)
        return sum(sizes[p] for p in parts), parts[0], f"{grp}-*.gguf ({len(parts)} parts)"
    return sizes[target], target, target


class VramApp(QWidget):
    # Signals so worker threads can update the UI on the GUI thread safely.
    _status = Signal(str, bool)            # (text, busy)
    _fetched = Signal(object, object)      # (sizes dict, names list)
    _result = Signal(str, str)             # (report text, status text)

    def __init__(self):
        super().__init__()
        self.sizes = {}                    # cached file list for the current repo
        self._repo_of_sizes = None
        self.setWindowTitle("llama.cpp GGUF VRAM estimator")
        self.setMinimumSize(720, 600)

        self._status.connect(self._on_status)
        self._fetched.connect(self._on_fetched)
        self._result.connect(self._on_result)

        self._build_ui()

    # -- UI construction -------------------------------------------------
    def _build_ui(self):
        grid = QGridLayout(self)
        grid.setContentsMargins(14, 14, 14, 14)
        grid.setHorizontalSpacing(8)
        grid.setVerticalSpacing(8)
        grid.setColumnStretch(1, 1)

        r = 0
        grid.addWidget(QLabel("HF repo id"), r, 0)
        self.repo = QLineEdit("Qwen/Qwen3-8B-GGUF")
        grid.addWidget(self.repo, r, 1)
        self.fetch_btn = QPushButton("Fetch files")
        self.fetch_btn.clicked.connect(self.fetch_files)
        grid.addWidget(self.fetch_btn, r, 2)

        r += 1
        grid.addWidget(QLabel("GGUF file / quant"), r, 0)
        self.file = QComboBox()
        self.file.setEditable(True)
        self.file.setEditText("Q4_K_M")
        grid.addWidget(self.file, r, 1, 1, 2)

        # numeric fields -------------------------------------------------
        r += 1
        nums = QGridLayout()
        nums.setHorizontalSpacing(8)
        nums.setVerticalSpacing(6)
        for c in (1, 3):
            nums.setColumnStretch(c, 1)

        self.in_tok = self._spin(0, 1_000_000, 4096)
        self.out_tok = self._spin(0, 1_000_000, 512)
        self.parallel = self._spin(1, 4096, 1)
        self.gpus = self._spin(1, 256, 1)
        self.vram = QDoubleSpinBox()
        self.vram.setRange(0.5, 4096.0)
        self.vram.setDecimals(1)
        self.vram.setValue(32.0)
        self.cache = QComboBox()
        self.cache.addItems(list(core.CACHE_BYTES))
        self.cache.setCurrentText("f16")

        nums.addWidget(QLabel("Max input tok"), 0, 0);   nums.addWidget(self.in_tok, 0, 1)
        nums.addWidget(QLabel("Max output tok"), 0, 2);  nums.addWidget(self.out_tok, 0, 3)
        nums.addWidget(QLabel("Parallel slots"), 1, 0);  nums.addWidget(self.parallel, 1, 1)
        nums.addWidget(QLabel("KV cache type"), 1, 2);   nums.addWidget(self.cache, 1, 3)
        nums.addWidget(QLabel("GPUs"), 2, 0);            nums.addWidget(self.gpus, 2, 1)
        nums.addWidget(QLabel("VRAM/GPU (GiB)"), 2, 2);  nums.addWidget(self.vram, 2, 3)
        grid.addLayout(nums, r, 0, 1, 3)

        r += 1
        grid.addWidget(QLabel("HF token (optional)"), r, 0)
        self.token = QLineEdit(os.environ.get("HF_TOKEN", ""))
        self.token.setEchoMode(QLineEdit.Password)
        grid.addWidget(self.token, r, 1, 1, 2)

        # action + status ------------------------------------------------
        r += 1
        actions = QHBoxLayout()
        self.est_btn = QPushButton("Estimate")
        self.est_btn.clicked.connect(self.estimate)
        actions.addWidget(self.est_btn)
        self.status = QLabel("Ready")
        self.status.setStyleSheet("color: #777;")
        actions.addWidget(self.status, 1)
        grid.addLayout(actions, r, 0, 1, 3)

        # output ---------------------------------------------------------
        r += 1
        grid.setRowStretch(r, 1)
        self.out = QPlainTextEdit()
        self.out.setReadOnly(True)
        self.out.setLineWrapMode(QPlainTextEdit.NoWrap)
        mono = QFontDatabase.systemFont(QFontDatabase.FixedFont)
        mono.setPointSize(11)
        mono.setStyleHint(QFont.Monospace)
        self.out.setFont(mono)
        self.out.setStyleSheet(
            "QPlainTextEdit { background: #1e1e1e; color: #e6e6e6; border: 0; }")
        grid.addWidget(self.out, r, 0, 1, 3)

    def _spin(self, lo, hi, val):
        s = QSpinBox()
        s.setRange(lo, hi)
        s.setValue(val)
        return s

    # -- signal handlers (run on the GUI thread) -------------------------
    def _set_busy(self, busy):
        self.est_btn.setEnabled(not busy)
        self.fetch_btn.setEnabled(not busy)

    def _on_status(self, text, busy):
        self.status.setText(text)
        self._set_busy(busy)

    def _on_fetched(self, sizes, names):
        self.sizes = sizes
        self.file.clear()
        self.file.addItems(names)
        if names:
            self.file.setCurrentText(names[0])
        self.status.setText(f"{len(names)} GGUF file(s) found")
        self._set_busy(False)

    def _on_result(self, text, status):
        self.out.setPlainText(text)
        self.status.setText(status)
        self._set_busy(False)

    # -- actions (each spawns a worker thread) ---------------------------
    def fetch_files(self):
        repo = self.repo.text().strip()
        token = self.token.text().strip() or None
        if not repo:
            self._on_status("enter a repo id first", False)
            return
        self._on_status(f"listing files in {repo}…", True)
        threading.Thread(target=self._fetch_worker, args=(repo, token), daemon=True).start()

    def _fetch_worker(self, repo, token):
        try:
            sizes = dict(core.hf_list_gguf(repo, token))
            self._repo_of_sizes = repo
            self._fetched.emit(sizes, sorted(sizes))
        except urllib.error.HTTPError as e:
            self._status.emit(f"HTTP {e.code}: {e.reason}", False)
        except Exception as e:                                  # noqa: BLE001
            self._status.emit(f"error: {e}", False)

    def estimate(self):
        repo = self.repo.text().strip()
        if not repo:
            self._on_status("enter a repo id first", False)
            return
        args = dict(
            repo=repo,
            choice=self.file.currentText().strip(),
            in_tok=self.in_tok.value(),
            out_tok=self.out_tok.value(),
            parallel=self.parallel.value(),
            cache=self.cache.currentText(),
            gpus=self.gpus.value(),
            vram=self.vram.value(),
            token=self.token.text().strip() or None,
        )
        self._on_status("reading GGUF header…", True)
        threading.Thread(target=self._estimate_worker, args=(args,), daemon=True).start()

    def _estimate_worker(self, a):
        try:
            sizes = self.sizes
            if not sizes or a["repo"] != self._repo_of_sizes:
                sizes = dict(core.hf_list_gguf(a["repo"], a["token"]))
                self.sizes = sizes
                self._repo_of_sizes = a["repo"]
            if not sizes:
                raise ValueError("no .gguf files in this repo")

            weight_bytes, meta_file, shown = resolve_target(sizes, a["choice"])
            url = f"{core.HF}/{a['repo']}/resolve/main/{urllib.parse.quote(meta_file)}"
            md = core.parse_gguf_header(core.make_range_fetch(url, a["token"]))
            est = core.estimate(md, weight_bytes, a["in_tok"], a["out_tok"],
                                a["parallel"], a["cache"], a["gpus"], a["vram"])
            text = core.report(est, shown, a["cache"])
            fits = est["per_gpu"] <= est["vram_gb"] * core.GIB
            self._result.emit(text, "done — " + ("fits ✓" if fits else "does NOT fit ✗"))
        except urllib.error.HTTPError as e:
            self._status.emit(f"HTTP {e.code}: {e.reason}", False)
        except Exception as e:                                  # noqa: BLE001
            self._status.emit(f"error: {e}", False)


def main():
    app = QApplication(sys.argv)
    win = VramApp()
    win.show()
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
