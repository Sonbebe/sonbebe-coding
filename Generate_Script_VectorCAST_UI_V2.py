import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
import threading
import queue
import logging
import os

from TC_generator_v2 import VectorCastGenerator, logger


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
        self.include_dirs_var = tk.StringVar()
        self.unit_name_var = tk.StringVar()
        self.env_name_var = tk.StringVar(value="TEST_SV")
        self.output_file_var = tk.StringVar(value="Result_Final.tst")
        self.verbose_var = tk.BooleanVar(value=False)
        self.is_cpp_var = tk.BooleanVar(value=False)

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
        ttk.Label(main_frame, text="C Source (File/Folder):").grid(
            row=0, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.c_file_var).grid(
            row=0, column=1, sticky=tk.EW, padx=5
        )
        btn_frame = ttk.Frame(main_frame)
        btn_frame.grid(row=0, column=2)
        ttk.Button(btn_frame, text="Select Folder", command=self._browse_c_folder).pack(side=tk.LEFT, padx=1)

        # Row 1: Include Dirs
        ttk.Label(main_frame, text="Include Paths (;):").grid(
            row=1, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.include_dirs_var).grid(
            row=1, column=1, sticky=tk.EW, padx=5
        )
        ttk.Button(main_frame, text="Select Folder", command=self._browse_include_dir).grid(
            row=1, column=2
        )

        # Row 2: Unit Name
        ttk.Label(main_frame, text="Unit Name (UUT):").grid(
            row=2, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.unit_name_var).grid(
            row=2, column=1, sticky=tk.EW, padx=5
        )

        # Row 3: Environment Name
        ttk.Label(main_frame, text="Environment Name:").grid(
            row=3, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.env_name_var).grid(
            row=3, column=1, sticky=tk.EW, padx=5
        )

        # Row 4: Output File
        ttk.Label(main_frame, text="Output Filename:").grid(
            row=4, column=0, sticky=tk.W, pady=5
        )
        ttk.Entry(main_frame, textvariable=self.output_file_var).grid(
            row=4, column=1, sticky=tk.EW, padx=5
        )

        # Row 5: Options
        ttk.Checkbutton(
            main_frame, text="Verbose Logging (Debug)", variable=self.verbose_var
        ).grid(row=5, column=0, sticky=tk.W, pady=10)

        ttk.Checkbutton(
            main_frame, text="C++ Mode", variable=self.is_cpp_var
        ).grid(row=5, column=1, sticky=tk.W, pady=10)

        # Row 6: Run Button
        self.run_btn = ttk.Button(
            main_frame, text="GENERATE SCRIPT", command=self._start_generation_thread
        )
        self.run_btn.grid(row=6, column=0, columnspan=3, pady=10, sticky=tk.EW)

        # Row 7: Log Area
        ttk.Label(main_frame, text="Execution Log:").grid(row=7, column=0, sticky=tk.W)
        self.log_area = scrolledtext.ScrolledText(
            main_frame, height=15, state="disabled"
        )
        self.log_area.grid(row=8, column=0, columnspan=3, sticky=tk.NSEW, pady=5)
        main_frame.rowconfigure(8, weight=1)

    def _browse_path(self):
        is_folder = messagebox.askyesno("Select Type", "Do you want to select a Folder?\n(Click 'No' to select a File)")
        if is_folder:
            path = filedialog.askdirectory()
        else:
            path = filedialog.askopenfilename(
                filetypes=[("C/C++ Files", "*.c *.cpp *.cxx *.cc"), ("All Files", "*.*")]
            )

        if path:
            self.c_file_var.set(path)
            if os.path.isdir(path):
                self.unit_name_var.set("<Auto from filename>")
                self.output_file_var.set("<Auto from filename>")
            else:
                basename = os.path.basename(path)
                unit_guess = os.path.splitext(basename)[0]
                if not self.unit_name_var.get() or self.unit_name_var.get() == "<Auto from filename>":
                    self.unit_name_var.set(unit_guess)
                if not self.output_file_var.get() or self.output_file_var.get() == "<Auto from filename>":
                    self.output_file_var.set(f"Result_{unit_guess}.tst")

    def _browse_c_folder(self):
        directory = filedialog.askdirectory()
        if directory:
            self.c_file_var.set(directory)
            self.unit_name_var.set("<Auto from filename>")
            self.output_file_var.set("<Auto from filename>")

    def _browse_include_dir(self):
        directory = filedialog.askdirectory()
        if directory:
            current = self.include_dirs_var.get()
            if current:
                self.include_dirs_var.set(current + ";" + directory)
            else:
                self.include_dirs_var.set(directory)

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
        input_path = self.c_file_var.get()
        include_dirs_str = self.include_dirs_var.get()
        base_env_name = self.env_name_var.get()
        verbose = self.verbose_var.get()
        is_cpp = self.is_cpp_var.get()
        
        include_dirs = [d.strip() for d in include_dirs_str.split(';') if d.strip()]

        if not input_path:
            logger.error("Missing required field: C Source.")
            self.root.after(0, lambda: self.run_btn.config(state="normal"))
            return

        if verbose:
            logger.setLevel(logging.DEBUG)
        else:
            logger.setLevel(logging.INFO)

        try:
            if os.path.isdir(input_path):
                c_files = []
                for root_dir, dirs, files in os.walk(input_path):
                    for file in files:
                        if file.endswith(('.c', '.cpp', '.cxx', '.cc')):
                            c_files.append(os.path.join(root_dir, file))
            else:
                c_files = [input_path]

            if not c_files:
                logger.warning("No C/C++ files found in the specified path.")
                return

            for c_file in c_files:
                basename = os.path.basename(c_file)
                name_without_ext = os.path.splitext(basename)[0]
                
                if os.path.isdir(input_path):
                    unit_name = name_without_ext
                    output_file = f"Result_{unit_name}.tst"
                    env_name = f"{base_env_name}_{unit_name}"
                else:
                    unit_name = self.unit_name_var.get() or name_without_ext
                    output_file = self.output_file_var.get() or f"Result_{unit_name}.tst"
                    env_name = base_env_name

                logger.info(f"Processing: {c_file} -> Unit: {unit_name}")
                generator = VectorCastGenerator(c_file, unit_name, env_name, include_dirs, is_cpp=is_cpp)
                generator.run(output_file)
                
            messagebox.showinfo("Success", f"Script(s) generated successfully for {len(c_files)} file(s).")
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
