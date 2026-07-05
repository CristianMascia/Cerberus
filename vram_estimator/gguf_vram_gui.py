#!/usr/bin/env python3
"""
Simple desktop GUI for gguf_vram.py (the llama.cpp VRAM estimator).

Requires only the Python standard library (tkinter). Put this file next to
gguf_vram.py and run:

    python gguf_vram_gui.py

Enter a Hugging Face repo id, optionally click "Fetch files" to list the
available .gguf files, set your token budget, and press "Estimate".
Network work runs in a background thread so the window never freezes.
"""

import os
import sys
import threading
import tkinter as tk
import urllib.error
import urllib.parse
from tkinter import ttk

import gguf_vram as core


# --- HiDPI / cross-platform display helpers --------------------------------
def enable_windows_dpi_awareness():
    """On Windows, opt into per-monitor DPI so Tk renders crisply instead of
    being bitmap-upscaled (which looks blurry / low-res). No-op elsewhere.

    Must be called *before* creating the Tk root.
    """
    if sys.platform != "win32":
        return
    import ctypes
    try:                                   # Windows 8.1+  (PER_MONITOR_AWARE)
        ctypes.windll.shcore.SetProcessDpiAwareness(2)
    except (AttributeError, OSError):
        try:                               # Vista/7/8  (system-DPI aware)
            ctypes.windll.user32.SetProcessDPIAware()
        except (AttributeError, OSError):
            pass


# Named Tk fonts shared by every ttk widget. We resize these directly because
# on Linux they are often defined in *pixels* (negative size), which the
# 'tk scaling' factor does NOT affect — hence "scaling" alone changes nothing.
_SCALABLE_FONTS = (
    "TkDefaultFont", "TkTextFont", "TkFixedFont", "TkMenuFont",
    "TkHeadingFont", "TkCaptionFont", "TkSmallCaptionFont",
    "TkTooltipFont", "TkIconFont",
)


def detect_scale(root):
    """Best-effort UI scale factor. VRAM_GUI_SCALE overrides everything;
    otherwise take the larger of the logical DPI and the physical DPI derived
    from the screen's mm size (X often reports 96 logical even on HiDPI)."""
    override = os.environ.get("VRAM_GUI_SCALE", "").strip()
    if override:
        try:
            return max(0.5, float(override))
        except ValueError:
            pass
    try:
        dpi_logical = root.winfo_fpixels("1i")
    except tk.TclError:
        dpi_logical = 96.0
    try:
        mm = root.winfo_screenmmwidth()
        dpi_physical = root.winfo_screenwidth() / (mm / 25.4) if mm else 96.0
    except tk.TclError:
        dpi_physical = 96.0
    dpi = max(dpi_logical, dpi_physical)
    return max(1.0, round(dpi / 96.0, 2))


class Scaler:
    """Live UI zoom: enlarges the named Tk fonts (and any custom fonts we
    register) by a factor, and keeps 'tk scaling' in sync for widget geometry.
    Works on Linux/macOS/Windows regardless of DPI reporting."""

    def __init__(self, root, factor):
        self.root = root
        from tkinter import font as tkfont
        self._tkfont = tkfont
        self._bases = []                       # list of (Font, original size)
        for name in _SCALABLE_FONTS:
            try:
                f = tkfont.nametofont(name)
            except tk.TclError:
                continue
            self._bases.append((f, f.cget("size")))
        self.apply(factor)

    def register(self, font):
        """Add a custom Font so it zooms together with the rest of the UI."""
        base = font.cget("size")
        self._bases.append((font, base))
        self._apply_one(font, base)
        return font

    def _apply_one(self, font, base):
        if not base:
            return
        sign = -1 if base < 0 else 1
        font.configure(size=sign * max(6, int(round(abs(base) * self.factor))))

    def apply(self, factor):
        self.factor = max(0.6, float(factor))
        for font, base in self._bases:
            self._apply_one(font, base)
        self.root.tk.call("tk", "scaling", self.factor * 96.0 / 72.0)


def mono_font():
    """Pick a monospace family that actually exists on this platform."""
    if sys.platform == "darwin":
        candidates = ("Menlo", "Monaco", "Courier")
    elif sys.platform == "win32":
        candidates = ("Consolas", "Cascadia Mono", "Courier New")
    else:
        candidates = ("DejaVu Sans Mono", "Liberation Mono", "Noto Sans Mono",
                      "Ubuntu Mono", "monospace")
    for name in candidates:
        if _has_font(name):
            return name
    return "Courier"


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


class App:
    def __init__(self, root):
        self.root = root
        self.sizes = {}            # cached file list for the current repo
        root.title("llama.cpp GGUF VRAM estimator")

        style = ttk.Style()
        try:
            style.theme_use("clam")
        except tk.TclError:
            pass

        # HiDPI: enlarge every font by a live-adjustable factor.
        self.scaler = Scaler(root, detect_scale(root))

        # Force one explicit UI font onto every ttk widget. Themes like 'clam'
        # don't always honour the named default fonts, so resizing those alone
        # leaves buttons/entries tiny — configuring the root style '.' does the
        # trick, and mutating this Font live re-renders all of them.
        from tkinter import font as tkfont
        _base = tkfont.nametofont("TkDefaultFont")
        self.ui_font = self.scaler.register(
            tkfont.Font(family=_base.cget("family"), size=_base.cget("size") or 10))
        style.configure(".", font=self.ui_font)
        root.option_add("*Font", self.ui_font)
        root.option_add("*TCombobox*Listbox.font", self.ui_font)

        root.minsize(int(680 * self.scaler.factor), int(560 * self.scaler.factor))
        root.bind("<Control-plus>", lambda e: self.zoom(+0.1))
        root.bind("<Control-equal>", lambda e: self.zoom(+0.1))
        root.bind("<Control-minus>", lambda e: self.zoom(-0.1))
        root.bind("<Control-0>", lambda e: self.set_zoom(1.0))

        pad = {"padx": 6, "pady": 4}
        frm = ttk.Frame(root, padding=12)
        frm.grid(sticky="nsew")
        root.columnconfigure(0, weight=1)
        root.rowconfigure(0, weight=1)
        frm.columnconfigure(1, weight=1)

        r = 0
        ttk.Label(frm, text="HF repo id").grid(row=r, column=0, sticky="w", **pad)
        self.repo = ttk.Entry(frm)
        self.repo.grid(row=r, column=1, sticky="ew", **pad)
        self.repo.insert(0, "Qwen/Qwen3-8B-GGUF")
        ttk.Button(frm, text="Fetch files", command=self.fetch_files)\
            .grid(row=r, column=2, sticky="ew", **pad)

        r += 1
        ttk.Label(frm, text="GGUF file / quant").grid(row=r, column=0, sticky="w", **pad)
        self.file = ttk.Combobox(frm, values=[])
        self.file.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        self.file.set("Q4_K_M")

        # numeric fields -------------------------------------------------
        grid2 = ttk.Frame(frm)
        r += 1
        grid2.grid(row=r, column=0, columnspan=3, sticky="ew", pady=(6, 0))
        for i in range(4):
            grid2.columnconfigure(i, weight=1)

        def field(parent, label, default, col, row):
            ttk.Label(parent, text=label).grid(row=row, column=col, sticky="w", padx=6, pady=4)
            e = ttk.Entry(parent, width=10)
            e.grid(row=row, column=col + 1, sticky="ew", padx=6, pady=4)
            e.insert(0, str(default))
            return e

        self.in_tok = field(grid2, "Max input tok", 4096, 0, 0)
        self.out_tok = field(grid2, "Max output tok", 512, 2, 0)
        self.parallel = field(grid2, "Parallel slots", 1, 0, 1)

        ttk.Label(grid2, text="KV cache type").grid(row=1, column=2, sticky="w", padx=6, pady=4)
        self.cache = ttk.Combobox(grid2, values=list(core.CACHE_BYTES), width=8, state="readonly")
        self.cache.set("f16")
        self.cache.grid(row=1, column=3, sticky="ew", padx=6, pady=4)

        self.gpus = field(grid2, "GPUs", 1, 0, 2)
        self.vram = field(grid2, "VRAM/GPU (GiB)", 32, 2, 2)

        r += 1
        ttk.Label(frm, text="HF token (optional)").grid(row=r, column=0, sticky="w", **pad)
        self.token = ttk.Entry(frm, show="*")
        self.token.grid(row=r, column=1, columnspan=2, sticky="ew", **pad)
        self.token.insert(0, os.environ.get("HF_TOKEN", ""))

        # action + status ------------------------------------------------
        r += 1
        self.btn = ttk.Button(frm, text="Estimate", command=self.estimate)
        self.btn.grid(row=r, column=0, sticky="ew", **pad)
        self.status = ttk.Label(frm, text="Ready", foreground="#555")
        self.status.grid(row=r, column=1, sticky="w", **pad)

        zoom = ttk.Frame(frm)
        zoom.grid(row=r, column=2, sticky="e", **pad)
        ttk.Label(zoom, text="Zoom").pack(side="left", padx=(0, 4))
        self.zoom_box = ttk.Combobox(
            zoom, width=6, state="readonly",
            values=["100%", "125%", "150%", "175%", "200%", "250%"])
        self.zoom_box.set(f"{round(self.scaler.factor * 100)}%")
        self.zoom_box.pack(side="left")
        self.zoom_box.bind("<<ComboboxSelected>>",
                           lambda e: self.set_zoom(int(self.zoom_box.get().rstrip("%")) / 100))

        # output ---------------------------------------------------------
        r += 1
        frm.rowconfigure(r, weight=1)
        from tkinter import font as tkfont
        out_font = self.scaler.register(tkfont.Font(family=mono_font(), size=11))
        self.out = tk.Text(frm, height=16, wrap="none",
                           font=out_font,
                           background="#1e1e1e", foreground="#e6e6e6",
                           insertbackground="#e6e6e6", borderwidth=0)
        self.out.grid(row=r, column=0, columnspan=3, sticky="nsew", pady=(8, 0))
        self.out.configure(state="disabled")

    # -- helpers ---------------------------------------------------------
    def _set_status(self, text, busy=False):
        self.status.configure(text=text)
        self.btn.configure(state="disabled" if busy else "normal")

    def _write(self, text):
        self.out.configure(state="normal")
        self.out.delete("1.0", "end")
        self.out.insert("1.0", text)
        self.out.configure(state="disabled")

    # -- zoom ------------------------------------------------------------
    def set_zoom(self, factor):
        self.scaler.apply(factor)
        self.zoom_box.set(f"{round(self.scaler.factor * 100)}%")

    def zoom(self, delta):
        self.set_zoom(self.scaler.factor + delta)

    # -- actions (each spawns a worker thread) ---------------------------
    def fetch_files(self):
        repo = self.repo.get().strip()
        token = self.token.get().strip() or None
        if not repo:
            self._set_status("enter a repo id first")
            return
        self._set_status(f"listing files in {repo}…", busy=True)
        threading.Thread(target=self._fetch_worker, args=(repo, token), daemon=True).start()

    def _fetch_worker(self, repo, token):
        try:
            files = core.hf_list_gguf(repo, token)
            sizes = dict(files)
            names = sorted(sizes)
            self.root.after(0, lambda: self._fetch_done(sizes, names))
        except urllib.error.HTTPError as e:
            self.root.after(0, lambda: self._set_status(f"HTTP {e.code}: {e.reason}"))
        except Exception as e:                                  # noqa: BLE001
            self.root.after(0, lambda: self._set_status(f"error: {e}"))

    def _fetch_done(self, sizes, names):
        self.sizes = sizes
        self.file["values"] = names
        if names:
            self.file.set(names[0])
        self._set_status(f"{len(names)} GGUF file(s) found")

    def estimate(self):
        try:
            args = dict(
                repo=self.repo.get().strip(),
                choice=self.file.get().strip(),
                in_tok=int(self.in_tok.get()),
                out_tok=int(self.out_tok.get()),
                parallel=int(self.parallel.get()),
                cache=self.cache.get(),
                gpus=int(self.gpus.get()),
                vram=float(self.vram.get()),
                token=self.token.get().strip() or None,
            )
        except ValueError:
            self._set_status("check the numeric fields")
            return
        if not args["repo"]:
            self._set_status("enter a repo id first")
            return
        self._set_status("reading GGUF header…", busy=True)
        threading.Thread(target=self._estimate_worker, args=(args,), daemon=True).start()

    def _estimate_worker(self, a):
        try:
            sizes = self.sizes
            if not sizes or a["repo"] != getattr(self, "_repo_of_sizes", None):
                sizes = dict(core.hf_list_gguf(a["repo"], a["token"]))
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
            self.root.after(0, lambda: (self._write(text),
                                        self._set_status("done — " +
                                                         ("fits ✓" if fits else "does NOT fit ✗"))))
        except urllib.error.HTTPError as e:
            self.root.after(0, lambda: self._set_status(f"HTTP {e.code}: {e.reason}"))
        except Exception as e:                                  # noqa: BLE001
            self.root.after(0, lambda: self._set_status(f"error: {e}"))


def _has_font(name):
    try:
        from tkinter import font
        return name in font.families()
    except Exception:                                          # noqa: BLE001
        return False


def main():
    enable_windows_dpi_awareness()   # must run before Tk() to avoid blur
    root = tk.Tk()
    App(root)
    root.mainloop()


if __name__ == "__main__":
    main()
