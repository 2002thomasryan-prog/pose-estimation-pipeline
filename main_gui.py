"""Main Tkinter GUI for the biomechanical pose pipeline: keypoint extraction, scoring and report generation."""

import os
import queue
import sys
import threading
import tkinter as tk
from tkinter import filedialog, scrolledtext, ttk
import numpy as np

SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, SCRIPT_DIR)

from pose_extractor import SAM3D_AVAILABLE as _SAM3D_AVAILABLE


class PipelineGUI:

    MODELS = ["YOLOv8", "RTMPose"] + (["SAM-3D-Body"] if _SAM3D_AVAILABLE else [])
    SPORTS = ["Running", "Cycling", "Rowing"]

    def __init__(self, root: tk.Tk):
        self.root    = root
        self.q       = queue.Queue()
        self.running = False

        self.video_path = tk.StringVar()
        self.output_dir = tk.StringVar()
        self.model      = tk.StringVar(value=self.MODELS[0])
        self.sport      = tk.StringVar(value=self.SPORTS[0])

        self.last_scorer_data        = None
        self.last_kps_array          = None
        self.last_all_cycles         = None
        self.last_all_cycle_events   = None
        self.last_sport              = None
        self.last_annotated_video    = None

        self._build_ui()
        self.root.after(100, self._poll_queue)

    def _build_ui(self):
        self.root.title("Biomechanical Pose Estimation Pipeline")
        self.root.geometry("820x560")
        self.root.configure(bg="#1a1a2e")
        self.root.resizable(True, True)

        style = ttk.Style()
        style.theme_use("clam")
        style.configure("TFrame",       background="#1a1a2e")
        style.configure("TLabel",       background="#1a1a2e", foreground="#e0e0e0",
                         font=("Segoe UI", 10))
        style.configure("TButton",      font=("Segoe UI", 10, "bold"),
                         padding=6)
        style.configure("TCombobox",    font=("Segoe UI", 10))
        style.configure("Header.TLabel",background="#1a1a2e", foreground="#64b5f6",
                         font=("Segoe UI", 13, "bold"))

        root_frame = ttk.Frame(self.root, padding=16)
        root_frame.pack(fill=tk.BOTH, expand=True)

        ttk.Label(root_frame, text="⚙  Biomechanical Pose Estimation Pipeline",
                  style="Header.TLabel").grid(row=0, column=0, columnspan=3,
                                               sticky="w", pady=(0, 12))

        ttk.Label(root_frame, text="Video file:").grid(row=1, column=0, sticky="w")
        ttk.Entry(root_frame, textvariable=self.video_path, width=55).grid(
            row=1, column=1, padx=6, sticky="ew")
        ttk.Button(root_frame, text="Browse…", command=self._browse_video).grid(
            row=1, column=2)

        ttk.Label(root_frame, text="Output folder:").grid(row=2, column=0, sticky="w",
                                                           pady=(6, 0))
        ttk.Entry(root_frame, textvariable=self.output_dir, width=55).grid(
            row=2, column=1, padx=6, sticky="ew", pady=(6, 0))
        ttk.Button(root_frame, text="Browse…", command=self._browse_output).grid(
            row=2, column=2, pady=(6, 0))

        sel_frame = ttk.Frame(root_frame)
        sel_frame.grid(row=3, column=0, columnspan=3, sticky="w", pady=(10, 0))
        ttk.Label(sel_frame, text="Model:").pack(side=tk.LEFT)
        ttk.Combobox(sel_frame, textvariable=self.model, values=self.MODELS,
                     state="readonly", width=16).pack(side=tk.LEFT, padx=(4, 16))
        ttk.Label(sel_frame, text="Sport:").pack(side=tk.LEFT)
        ttk.Combobox(sel_frame, textvariable=self.sport, values=self.SPORTS,
                     state="readonly", width=12).pack(side=tk.LEFT, padx=4)

        btn_frame = ttk.Frame(root_frame)
        btn_frame.grid(row=4, column=0, columnspan=3, pady=(12, 6), sticky="ew")
        self.run_btn = ttk.Button(btn_frame, text="▶   RUN PIPELINE",
                                  command=self._run, width=22)
        self.run_btn.pack(side=tk.LEFT)
        self.progress = ttk.Progressbar(btn_frame, mode="indeterminate", length=200)
        self.progress.pack(side=tk.LEFT, padx=12)
        self.report_btn = ttk.Button(btn_frame, text="📄  Export Report",
                                     command=self._export_report, width=18,
                                     state="disabled")
        self.report_btn.pack(side=tk.LEFT, padx=(8, 0))
        self.video_btn = ttk.Button(btn_frame, text="🎬  Export Annotated Video",
                                    command=self._export_annotated_video, width=28,
                                    state="disabled")
        self.video_btn.pack(side=tk.LEFT, padx=(8, 0))

        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(root_frame, textvariable=self.status_var,
                  foreground="#aaaaaa").grid(row=5, column=0, columnspan=3,
                                             sticky="w", pady=(0, 4))

        self.log_text = scrolledtext.ScrolledText(
            root_frame, height=18, bg="#0d0d1a", fg="#c8e6c9",
            font=("Consolas", 9), insertbackground="white",
        )
        self.log_text.grid(row=6, column=0, columnspan=3, sticky="nsew", pady=(4, 0))

        root_frame.columnconfigure(1, weight=1)
        root_frame.rowconfigure(6, weight=1)

    def _browse_video(self):
        p = filedialog.askopenfilename(
            title="Select video file",
            filetypes=[
                ("Video files", "*.mp4 *.avi *.mov *.mkv *.MP4 *.AVI *.MOV"),
                ("All files", "*.*"),
            ],
        )
        if p:
            self.video_path.set(p)
            if not self.output_dir.get():
                self.output_dir.set(os.path.dirname(p))

    def _browse_output(self):
        p = filedialog.askdirectory(title="Select output folder")
        if p:
            self.output_dir.set(p)


    def _run(self):
        if not self.video_path.get():
            self._set_status("⚠  Please select a video file.")
            return
        if not os.path.isfile(self.video_path.get()):
            self._set_status("⚠  Video file not found — please re-select.")
            return
        if not self.output_dir.get():
            self._set_status("⚠  Please select an output folder.")
            return

        self.run_btn.config(state="disabled", text="Running…")
        self.progress.start(10)
        self.log_text.delete("1.0", tk.END)
        self.running = True

        threading.Thread(target=self._pipeline_thread, daemon=True).start()


    def _pipeline_thread(self):
        try:
            video  = self.video_path.get()
            outdir = self.output_dir.get()
            model  = self.model.get()
            sport  = self.sport.get().lower()
            stem   = os.path.splitext(os.path.basename(video))[0]
            safe_m = model.replace("-", "").replace(" ", "")

            self._log("=" * 62)
            self._log(f"  VIDEO : {os.path.basename(video)}")
            self._log(f"  MODEL : {model}")
            self._log(f"  SPORT : {sport.capitalize()}")
            self._log(f"  OUTPUT: {outdir}")
            self._log("=" * 62)

            self._log("\n[1/4] Loading pipeline modules...")
            self._set_status("Loading modules…")
            from pose_extractor import extract_keypoints

            if sport == 'running':
                from running_scorer import RunningScorer as ScorerClass
            elif sport == 'cycling':
                from cycling_scorer import CyclingScorer as ScorerClass
            elif sport == 'rowing':
                from rowing_scorer import RowingScorer as ScorerClass
            else:
                raise ValueError(f"Unknown sport '{sport}'. "
                                 f"Expected 'running', 'cycling', or 'rowing'.")

            self._log(f"      Modules loaded  ({ScorerClass.__name__}).")

            self._log(f"\n[2/4] Extracting keypoints  ({model})...")
            self._set_status(f"Extracting keypoints ({model})…")
            kps_array, fps, backend, frame_indices = extract_keypoints(
                video, model, progress_cb=self._log
            )
            self._log(
                f"      ✓  {len(kps_array)} frames  |  "
                f"shape={kps_array.shape}  |  backend={backend}"
            )

            self._log(f"\n[3/4] Scoring biomechanics  ({sport.capitalize()})...")
            self._set_status(f"Scoring ({sport})…")

            scorer = ScorerClass(
                kps_array, backend=backend, fps=fps,
                video_path=video, log_cb=self._log
            )
            scorer.preprocess().detect_side().detect_cycles().compute()
            np.save(os.path.join(outdir, f'{stem}_{safe_m}_raw_kp.npy'),    kps_array)
            _cleaned_kp = getattr(scorer, 'kp_arr', None)
            if _cleaned_kp is None:
                _cleaned_kp = getattr(scorer, 'kp_seq', None)
            np.save(os.path.join(outdir, f'{stem}_{safe_m}_cleaned_kp.npy'), _cleaned_kp)
            scorer.print_summary()
            if sport == 'running':
                from running_scorer.event_diagnostics import export_diagnostics
                export_diagnostics(scorer, output_dir=os.path.join(outdir, 'diagnostics'))
            scores = scorer.to_dict()
            self._log(
                f"      ✓  Total score: {scores['total_score']:.1f}/100  "
                f"|  {scores['n_cycles']} cycles detected"
            )

            self.last_scorer_data     = scores
            self.last_kps_array       = kps_array
            self.last_all_cycles       = getattr(scorer, 'all_cycles', scorer.cycles)
            self.last_all_cycle_events = getattr(
                scorer, 'all_cycle_events', getattr(scorer, 'cycle_events', None)
            )
            self.last_sport           = sport
            _ckp = getattr(scorer, 'kp_arr', None)
            if _ckp is None:
                _ckp = getattr(scorer, 'kp_seq', None)
            self._last_cleaned_kp     = _ckp

            scorer.video_path = video

            ann_path = os.path.join(outdir, f'{stem}_{safe_m}_annotated.mp4')
            self._log(f"\n[4/4] Rendering annotated video...")
            self._set_status("Rendering annotated video…")
            try:
                from annotated_video_writer import write_annotated_video
                write_annotated_video(
                    video_path  = video,
                    kp_arr      = _ckp,
                    scorer_data = scores,
                    output_path = ann_path,
                    progress_cb = self._log,
                )
                scorer.annotated_video_path = ann_path
                self.last_annotated_video   = ann_path
                self._log(f"      ✓  Annotated video: {os.path.basename(ann_path)}")
            except Exception as ann_err:
                self._log(f"      ⚠  Annotated video failed: {ann_err}")
                scorer.annotated_video_path = video

            scores = scorer.to_dict()
            self.last_scorer_data = scores

            if sport == 'running':
                self._log("\n[Dash] Launching interactive running dashboard...")
                self._set_status("Opening interactive dashboard…")
                try:
                    from running_scorer import running_dash
                    running_dash.set_data(scorer.to_dict())
                    running_dash.launch(open_browser=True)
                    self._log("      ✓  Dashboard running at http://localhost:8766")
                except Exception as dash_err:
                    self._log(f"      ⚠  Dashboard could not launch: {dash_err}")

            elif sport == 'rowing':
                self._log("\n[Dash] Launching interactive rowing dashboard...")
                self._set_status("Opening interactive dashboard…")
                try:
                    from rowing_scorer import rowing_dash
                    rowing_dash.set_data(scorer.to_dict())
                    rowing_dash.launch(open_browser=True)
                    self._log("      ✓  Dashboard running at http://localhost:8765")
                except Exception as dash_err:
                    self._log(f"      ⚠  Dashboard could not launch: {dash_err}")

            elif sport == 'cycling':
                self._log("\n[Dash] Launching interactive cycling dashboard...")
                self._set_status("Opening interactive dashboard…")
                try:
                    from cycling_scorer import cycling_dash
                    cycling_dash.set_data(scorer.to_dict())
                    cycling_dash.launch(open_browser=True)
                    self._log("      ✓  Dashboard running at http://localhost:8767")
                except Exception as dash_err:
                    self._log(f"      ⚠  Dashboard could not launch: {dash_err}")

            self._log("\n" + "=" * 62)
            self._log(f"  ✓  PIPELINE COMPLETE")
            self._log(f"  Outputs saved to:  {outdir}")
            self._log("=" * 62)
            self._set_status(f"✓  Done  —  outputs saved to {outdir}")

        except Exception as exc:
            import traceback
            self._log(f"\n✗  ERROR: {exc}")
            self._log(traceback.format_exc())
            self._set_status(f"✗  Error: {exc}")

        finally:
            self.running = False
            self.root.after(0, self._reset_run_btn)

    def _export_report(self):
        """Generate and open a PDF biomechanical report for the last run."""
        if self.last_scorer_data is None:
            tk.messagebox.showwarning("No data",
                                      "Run the pipeline first before exporting a report.")
            return

        from tkinter import filedialog
        import os

        video    = self.video_path.get()
        stem     = os.path.splitext(os.path.basename(video))[0] if video else 'report'
        outdir   = self.output_dir.get() or os.path.dirname(video) or '.'
        pipeline = self.last_scorer_data.get('backend', 'unknown').lower()
        default  = os.path.join(outdir, f'{stem}_{pipeline}_biomech_report.pdf')

        out_path = filedialog.asksaveasfilename(
            title="Save Report As",
            initialfile=os.path.basename(default),
            initialdir=outdir,
            defaultextension='.pdf',
            filetypes=[('PDF files', '*.pdf')],
        )
        if not out_path:
            return

        self.report_btn.config(state="disabled", text="Generating…")
        self.root.update_idletasks()

        try:
            from report_generator import generate_report
            generate_report(
                data        = self.last_scorer_data,
                keypoints   = self.last_kps_array,
                output_path = out_path,
            )
            self._log(f"\n[Report] PDF saved to {out_path}")
            import subprocess, sys
            if sys.platform.startswith('win'):
                os.startfile(out_path)
            elif sys.platform == 'darwin':
                subprocess.Popen(['open', out_path])
            else:
                subprocess.Popen(['xdg-open', out_path])

        except Exception as exc:
            import traceback
            tk.messagebox.showerror("Report Error",
                                    f"Could not generate report:\n{exc}")
            self._log(f"\n[Report] ERROR: {exc}")
            self._log(traceback.format_exc())
        finally:
            self.report_btn.config(state="normal", text="📄  Export Report")

    def _export_annotated_video(self):
        """Re-render (or re-save) the annotated video for the last pipeline run."""
        if self.last_scorer_data is None:
            tk.messagebox.showwarning("No data",
                                      "Run the pipeline first before exporting.")
            return

        video  = self.video_path.get()
        stem   = os.path.splitext(os.path.basename(video))[0] if video else 'output'
        model  = self.model.get()
        safe_m = model.replace("-", "").replace(" ", "")
        outdir = self.output_dir.get() or os.path.dirname(video) or '.'
        default_name = f'{stem}_{safe_m}_annotated.mp4'

        out_path = filedialog.asksaveasfilename(
            title="Save Annotated Video As",
            initialfile=default_name,
            initialdir=outdir,
            defaultextension='.mp4',
            filetypes=[('MP4 video', '*.mp4'), ('All files', '*.*')],
        )
        if not out_path:
            return

        self.video_btn.config(state="disabled", text="Rendering…")
        self.root.update_idletasks()

        def _render_thread():
            try:
                from annotated_video_writer import write_annotated_video
                kp_arr = getattr(self, '_last_cleaned_kp', None)
                if kp_arr is None:
                    kp_arr = self.last_kps_array
                write_annotated_video(
                    video_path  = video,
                    kp_arr      = kp_arr,
                    scorer_data = self.last_scorer_data,
                    output_path = out_path,
                    progress_cb = self._log,
                )
                self.last_annotated_video = out_path
                self._log(f"\n[Video] Annotated video saved to {out_path}")
                import subprocess, sys as _sys
                if _sys.platform.startswith('win'):
                    os.startfile(out_path)
                elif _sys.platform == 'darwin':
                    subprocess.Popen(['open', out_path])
                else:
                    subprocess.Popen(['xdg-open', out_path])
            except Exception as exc:
                import traceback
                tk.messagebox.showerror("Video Export Error",
                                        f"Could not render annotated video:\n{exc}")
                self._log(f"\n[Video] ERROR: {exc}")
                self._log(traceback.format_exc())
            finally:
                self.root.after(0, lambda: self.video_btn.config(
                    state="normal", text="🎬  Export Annotated Video"))

        threading.Thread(target=_render_thread, daemon=True).start()

    def _reset_run_btn(self):
        self.progress.stop()
        self.run_btn.config(state="normal", text="▶   RUN PIPELINE")
        if self.last_scorer_data is not None:
            self.report_btn.config(state="normal")
            self.video_btn.config(state="normal")


    def _log(self, msg: str):
        self.q.put(str(msg))

    def _set_status(self, msg: str):
        self.q.put(f"__STATUS__{msg}")

    def _poll_queue(self):
        """Called every 100 ms to flush log messages from background thread."""
        try:
            while True:
                msg = self.q.get_nowait()
                if msg.startswith("__STATUS__"):
                    self.status_var.set(msg[10:])
                else:
                    self.log_text.insert(tk.END, msg + "\n")
                    self.log_text.see(tk.END)
        except queue.Empty:
            pass
        self.root.after(100, self._poll_queue)


def main():
    root = tk.Tk()
    PipelineGUI(root)
    root.mainloop()


if __name__ == "__main__":
    main()
