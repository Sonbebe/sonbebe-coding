import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import queue
import logging
import os

from Generate_Script_VectorCAST import VectorCastGenerator, logger


class QueueHandler(logging.Handler):
    """Class to redirect logging to a queue, which can be read by Tkinter."""

    def __init__(self, log_queue):
        super().__init__()
        self.log_queue = log_queue

    def emit(self, record):
        self.log_queue.put(self.format(record))


class VectorCastGeneratorUI:
    def __init__(self, root):
        self.root = root
        self.root.title("VectorCAST Script Generator (Professional Edition)")
        self.root.geometry("700x600")

        # Variables
        self.c_file_var = tk.StringVar()
        self.unit_name_var = tk.StringVar()
        self.env_name_var = tk.StringVar(value="TEST_SV")
        self.output_file_var = tk.StringVar(value="Result_Final.tst")
        self.verbose_var = tk.BooleanVar(value=False)

        # Layout
        self._create_widgets()

        # Logging Setup for UI
        self.log_queue = queue.Queue()
        self.queue_handler = QueueHandler(self.log_queue)
        self.queue_handler.setFormatter(
            logging.Formatter("%(asctime)s - [%(levelname)s] - %(message)s")
        )

        # Add handler to the logger from Generate_Script_VectorCAST
        logger.addHandler(self.queue_handler)

        # Start polling log queue
        self.root.after(100, self._poll_log_queue)

    def _create_widgets(self):
        # Main Frame
        main_frame = ttk.Frame(self.root, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Grid Config
        main_frame.columnconfigure(1, weight=1)

        # Row 0: C File
        ttk.Label(main_frame, text="C Source File:").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.c_file_var).grid(
            row=0, column=1, sticky=tk.EW, padx=5
        )
        ttk.Button(main_frame, text="Browse...", command=self._browse_c_file).grid(
            row=0, column=2
        )

        # Row 1: Unit Name
        ttk.Label(main_frame, text="Unit Name (UUT):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.unit_name_var).grid(
            row=1, column=1, sticky=tk.EW, padx=5
        )

        # Row 2: Environment Name
        ttk.Label(main_frame, text="Environment Name:").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.env_name_var).grid(
            row=2, column=1, sticky=tk.EW, padx=5
        )

        # Row 3: Output File
        ttk.Label(main_frame, text="Output Filename:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.output_file_var).grid(
            row=3, column=1, sticky=tk.EW, padx=5
        )

        # Row 4: Options
        ttk.Checkbutton(
            main_frame, text="Verbose Logging (Debug)", variable=self.verbose_var
        ).grid(row=4, column=0, columnspan=2, sticky=tk.W, pady=10)

        # Row 5: Run Button
        self.run_btn = ttk.Button(
            main_frame, text="GENERATE SCRIPT", command=self._start_generation_thread
        )
        self.run_btn.grid(row=5, column=0, columnspan=3, pady=10, sticky=tk.EW)

        # Row 6: Log Area
        ttk.Label(main_frame, text="Execution Log:").grid(row=6, column=0, sticky=tk.W)
        self.log_area = scrolledtext.ScrolledText(
            main_frame, height=15, state="disabled"
        )
        self.log_area.grid(row=7, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        main_frame.rowconfigure(7, weight=1)

    def _browse_c_file(self):
        filename = filedialog.askopenfilename(
            filetypes=[("C Files", "*.c"), ("All Files", "*.*")]
        )
        if filename:
            self.c_file_var.set(filename)
            # Auto-guess unit name from filename
            basename = os.path.basename(filename)
            unit_guess = os.path.splitext(basename)[0]
            if not self.unit_name_var.get():
                self.unit_name_var.set(unit_guess)

    def _poll_log_queue(self):
        while True:
            try:
                record = self.log_queue.get(block=False)
                self.log_area.configure(state="normal")
                self.log_area.insert(tk.END, record + "\n")
                self.log_area.see(tk.END)
                self.log_area.configure(state="disabled")
            except queue.Empty:
                break
        self.root.after(100, self._poll_log_queue)

    def _start_generation_thread(self):
        self.run_btn.config(state="disabled")
        self.log_area.configure(state="normal")
        self.log_area.delete(1.0, tk.END)
        self.log_area.configure(state="disabled")

        thread = threading.Thread(target=self._run_generation)
        thread.daemon = True
        thread.start()

    def _run_generation(self):
        c_file = self.c_file_var.get()
        unit_name = self.unit_name_var.get()
        env_name = self.env_name_var.get()
        output_file = self.output_file_var.get()
        verbose = self.verbose_var.get()

        if not c_file or not unit_name:
            logger.error("Missing required fields: C File or Unit Name.")
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            return

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        try:
            generator = VectorCastGenerator(c_file, unit_name, env_name)
            generator.run(output_file)
            messagebox.showinfo(
                "Success", f"Script generated successfully:\n{output_file}"
            )
        except Exception as e:
            logger.exception("An error occurred during generation.")
            messagebox.showerror("Error", f"An error occurred:\n{str(e)}")
        finally:
            self.root.after(0, lambda: self.run_btn.config(state="normal"))


def launch_ui():
    root = tk.Tk()
    app = VectorCastGeneratorUI(root)
    root.mainloop()


if __name__ == "__main__":
    launch_ui()

