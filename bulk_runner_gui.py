"""Batch runner: processes every sagittal video in a participant folder tree with RTMPose, saving keypoints and a PDF report per video."""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

VIDEO_EXTS = {'.mp4', '.avi', '.mov', '.mkv'}
MODEL      = "RTMPose"
SAFE_MODEL = "RTMPose"


def detect_sport(filename: str) -> str | None:
    """Return 'running' | 'cycling' | 'rowing' from filename prefix, or None."""
    name = os.path.basename(filename).lower()
    if name.startswith('running'):
        return 'running'
    if name.startswith('cycling'):
        return 'cycling'
    if name.startswith('rowing'):
        return 'rowing'
    return None


def is_sagittal(filename: str) -> bool:
    """True if filename contains 'sag' (case-insensitive) and NOT '45'."""
    name = os.path.basename(filename).lower()
    return 'sag' in name and '45' not in name


def collect_videos(participant_dir: str) -> list[tuple[str, str]]:
    """
    Walk participant_dir recursively.
    Return list of (video_path, sport) for all sagittal videos with known sport.
    """
    results = []
    for root, _dirs, files in os.walk(participant_dir):
        for f in sorted(files):
            if os.path.splitext(f)[1].lower() not in VIDEO_EXTS:
                continue
            if not is_sagittal(f):
                continue
            sport = detect_sport(f)
            if sport is None:
                continue
            results.append((os.path.join(root, f), sport))
    return results


class BulkRunnerGUI:

    def __init__(self, root: tk.Tk):
        self.root    = root
        self.q       = queue.Queue()
        self.running = False
        self._stop   = threading.Event()

        self.input_dir  = tk.StringVar()
        self.output_dir = tk.StringVar()

        self._build_ui()
        self.root.after(100, self._poll_queue)


    def _build_ui(self):
        self.root.title("Bulk Biomechanical Pipeline Runner")
        self.root.geometry("860x620")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",        background="#1a1a2e")
        style.configure("TLabel",        background="#1a1a2e", foreground="#e0e0e0",
                         font=("Segoe UI", 10))
        style.configure("TButton",       font=("Segoe UI", 10, "bold"), padding=6)
        style.configure("TCombobox",     font=("Segoe UI", 10))
        style.configure("Header.TLabel", background="#1a1a2e", foreground="#64b5f6",
                         font=("Segoe UI", 13, "bold"))
        style.configure("Sub.TLabel",    background="#1a1a2e", foreground="#aaaaaa",
                         font=("Segoe UI", 9))

        f = ttk.Frame(self.root, padding=16)
        f.pack(fill=tk.BOTH, expand=True)

        ttk.Label(f, text="⚙  Bulk Biomechanical Pipeline Runner  (RTMPose · Sagittal only)",
                  style="Header.TLabel").grid(row=0, column=0, columnspan=3,
                                              sticky="w", pady=(0, 4))
        ttk.Label(f, text="Detects sport from filename prefix. Skips 45° views. "
                           "Saves .npy + PDF per video.",
                  style="Sub.TLabel").grid(row=1, column=0, columnspan=3,
                                           sticky="w", pady=(0, 10))

        ttk.Label(f, text="Participant folder:").grid(row=2, column=0, sticky="w")
        ttk.Entry(f, textvariable=self.input_dir, width=58).grid(
            row=2, column=1, padx=6, sticky="ew")
        ttk.Button(f, text="Browse…", command=self._browse_input).grid(row=2, column=2)

        ttk.Label(f, text="Output root:").grid(row=3, column=0, sticky="w", pady=(6, 0))
        ttk.Entry(f, textvariable=self.output_dir, width=58).grid(
            row=3, column=1, padx=6, sticky="ew", pady=(6, 0))
        ttk.Button(f, text="Browse…", command=self._browse_output).grid(
            row=3, column=2, pady=(6, 0))

        ttk.Button(f, text="🔍  Preview videos", command=self._preview).grid(
            row=4, column=0, columnspan=3, sticky="w", pady=(10, 0))

        btn_frame = ttk.Frame(f)
        btn_frame.grid(row=5, column=0, columnspan=3, pady=(8, 6), sticky="ew")
        self.run_btn = ttk.Button(btn_frame, text="▶   RUN ALL",
                                  command=self._run, width=18)
        self.run_btn.pack(side=tk.LEFT)
        self.stop_btn = ttk.Button(btn_frame, text="⏹  Stop",
                                   command=self._stop_run, width=12, state="disabled")
        self.stop_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.progress = ttk.Progressbar(btn_frame, mode="determinate", length=240)
        self.progress.pack(side=tk.LEFT, padx=12)
        self.progress_label = tk.StringVar(value="")
        ttk.Label(btn_frame, textvariable=self.progress_label,
                  background="#1a1a2e", foreground="#aaaaaa",
                  font=("Segoe UI", 9)).pack(side=tk.LEFT)

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(f, textvariable=self.status_var,
                  foreground="#aaaaaa").grid(row=6, column=0, columnspan=3,
                                             sticky="w", pady=(0, 4))

        self.log_text = scrolledtext.ScrolledText(
            f, height=20, bg="#0d0d1a", fg="#c8e6c9",
            font=("Consolas", 9), insertbackground="white",
        )
        self.log_text.grid(row=7, column=0, columnspan=3, sticky="nsew", pady=(4, 0))

        f.columnconfigure(1, weight=1)
        f.rowconfigure(7, weight=1)


    def _browse_input(self):
        p = filedialog.askdirectory(title="Select participant folder")
        if p:
            self.input_dir.set(p)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(p))

    def _browse_output(self):
        p = filedialog.askdirectory(title="Select output root folder")
        if p:
            self.output_dir.set(p)


    def _preview(self):
        d = self.input_dir.get()
        if not d or not os.path.isdir(d):
            self._set_status("⚠  Please select a valid participant folder first.")
            return
        videos = collect_videos(d)
        self.log_text.delete("1.0", tk.END)
        if not videos:
            self._log("No sagittal videos found in that folder.")
            return
        self._log(f"Found {len(videos)} sagittal video(s) to process:\n")
        for vp, sport in videos:
            self._log(f"  [{sport.upper():<8}]  {os.path.basename(vp)}")
        self._log(f"\nModel: RTMPose  |  Outputs: .npy + PDF report")


    def _run(self):
        if not self.input_dir.get() or not os.path.isdir(self.input_dir.get()):
            self._set_status("⚠  Please select a valid participant folder.")
            return
        if not self.output_dir.get():
            self._set_status("⚠  Please select an output root folder.")
            return

        self._stop.clear()
        self.run_btn.config(state="disabled")
        self.stop_btn.config(state="normal")
        self.log_text.delete("1.0", tk.END)
        self.running = True

        threading.Thread(target=self._batch_thread, daemon=True).start()

    def _stop_run(self):
        self._stop.set()
        self._set_status("⏹  Stop requested — finishing current video…")
        self.stop_btn.config(state="disabled")


    def _batch_thread(self):
        try:
            input_dir  = self.input_dir.get()
            output_root = self.output_dir.get()

            videos = collect_videos(input_dir)
            if not videos:
                self._log("No sagittal videos found. Check folder and filenames.")
                return

            total = len(videos)
            self._log(f"{'='*62}")
            self._log(f"  BULK RUN  —  {total} video(s)  |  Model: RTMPose")
            self._log(f"  Input:   {input_dir}")
            self._log(f"  Output:  {output_root}")
            self._log(f"{'='*62}\n")

            participant = os.path.basename(os.path.normpath(input_dir))
            out_participant = os.path.join(output_root, participant)
            os.makedirs(out_participant, exist_ok=True)

            succeeded, failed, skipped = 0, 0, 0

            for idx, (video_path, sport) in enumerate(videos, start=1):
                if self._stop.is_set():
                    self._log(f"\n⏹  Stopped by user after {idx-1}/{total} videos.")
                    break

                self._set_progress(idx - 1, total)
                self._set_progress_label(f"{idx}/{total}")
                stem = os.path.splitext(os.path.basename(video_path))[0]

                self._log(f"\n{'─'*62}")
                self._log(f"  [{idx}/{total}]  {os.path.basename(video_path)}")
                self._log(f"  Sport: {sport.capitalize()}  |  Output: {out_participant}")
                self._log(f"{'─'*62}")

                try:
                    self._run_single(video_path, sport, stem, out_participant)
                    succeeded += 1
                except Exception as exc:
                    import traceback
                    self._log(f"\n  ✗  ERROR on {os.path.basename(video_path)}:")
                    self._log(f"     {exc}")
                    self._log(traceback.format_exc())
                    failed += 1

            self._set_progress(total, total)
            self._set_progress_label(f"{total}/{total}")

            self._log(f"\n{'='*62}")
            self._log(f"  BATCH COMPLETE")
            self._log(f"  ✓ Succeeded: {succeeded}   ✗ Failed: {failed}")
            self._log(f"  Outputs saved to: {out_participant}")
            self._log(f"{'='*62}")
            self._set_status(f"✓  Done  —  {succeeded}/{total} succeeded, {failed} failed")

        except Exception as exc:
            import traceback
            self._log(f"\n✗  BATCH ERROR: {exc}")
            self._log(traceback.format_exc())
            self._set_status(f"✗  Batch error: {exc}")

        finally:
            self.running = False
            self.root.after(0, self._reset_buttons)

    def _run_single(self, video_path: str, sport: str, stem: str, outdir: str):
        """Run the full pipeline for one video (pose → score → npy + PDF)."""

        self._log("  [1/3] Loading pipeline modules…")
        from pose_extractor import extract_keypoints

        if sport == 'running':
            from running_scorer import RunningScorer as ScorerClass
        elif sport == 'cycling':
            from cycling_scorer import CyclingScorer as ScorerClass
        elif sport == 'rowing':
            from rowing_scorer import RowingScorer as ScorerClass
        else:
            raise ValueError(f"Unknown sport '{sport}'")

        self._log(f"  [2/3] Extracting keypoints (RTMPose)…")
        self._set_status(f"Extracting: {os.path.basename(video_path)}")

        kps_array, fps, backend, frame_indices = extract_keypoints(
            video_path, MODEL, progress_cb=self._log
        )
        self._log(f"       ✓  {len(kps_array)} frames | shape={kps_array.shape} | backend={backend}")

        self._log(f"  [3/3] Scoring ({sport.capitalize()})…")
        self._set_status(f"Scoring: {os.path.basename(video_path)}")

        scorer = ScorerClass(
            kps_array, backend=backend, fps=fps,
            video_path=video_path, log_cb=self._log
        )
        scorer.preprocess().detect_side().detect_cycles().compute()

        np.save(os.path.join(outdir, f'{stem}_{SAFE_MODEL}_raw_kp.npy'),     kps_array)
        np.save(os.path.join(outdir, f'{stem}_{SAFE_MODEL}_cleaned_kp.npy'), scorer.kp_arr)
        self._log(f"       ✓  Keypoints saved")

        if sport == 'running':
            from running_scorer.event_diagnostics import export_diagnostics
            export_diagnostics(scorer, output_dir=os.path.join(outdir, 'diagnostics'))
            self._log(f"       ✓  Diagnostics saved")

        scorer_data = scorer.to_dict()
        report_path = os.path.join(outdir, f'{stem}_biomech_report.pdf')
        self._log(f"       Generating PDF report…")
        from report_generator import generate_report
        generate_report(
            data        = scorer_data,
            keypoints   = kps_array,
            output_path = report_path,
        )
        self._log(f"       ✓  Report saved: {os.path.basename(report_path)}")

        scores = scorer_data
        self._log(
            f"  ✓  Score: {scores['total_score']:.1f}/100  "
            f"|  {scores['n_cycles']} cycles  |  {sport.capitalize()}"
        )


    def _set_progress(self, value: int, maximum: int):
        pct = int(value / maximum * 100) if maximum > 0 else 0
        self.q.put(f"__PROGRESS__{pct}")

    def _set_progress_label(self, text: str):
        self.q.put(f"__PROGLABEL__{text}")

    def _reset_buttons(self):
        self.progress.stop()
        self.run_btn.config(state="normal")
        self.stop_btn.config(state="disabled")


    def _log(self, msg: str):
        self.q.put(str(msg))

    def _set_status(self, msg: str):
        self.q.put(f"__STATUS__{msg}")

    def _poll_queue(self):
        try:
            while True:
                msg = self.q.get_nowait()
                if msg.startswith("__STATUS__"):
                    self.status_var.set(msg[10:])
                elif msg.startswith("__PROGRESS__"):
                    self.progress["value"] = int(msg[12:])
                elif msg.startswith("__PROGLABEL__"):
                    self.progress_label.set(msg[13:])
                else:
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


def main():
    root = tk.Tk()
    BulkRunnerGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
