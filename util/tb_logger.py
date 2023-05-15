import os
import subprocess
import tensorboardX
import time

import util.logger as logger

class TBLogger(logger.Logger):
    MISC_TAG = "Misc"

    def __init__(self, run_tb=False):
        super().__init__()

        self._writer = None
        self._step_var_key = None
        self._collections = dict()

        self._run_tb = run_tb
        self._tb_proc = None

        return

    def __del__(self):
        if (self._tb_proc is not None):
            self._tb_proc.kill()
        return

    def reset(self):
        super().reset()
        return

    def configure_output_file(self, filename=None):
        super().configure_output_file(filename)

        output_dir = os.path.dirname(filename)
        self._delete_event_files(output_dir)
        self._writer = tensorboardX.SummaryWriter(output_dir)

        if (self._run_tb):
            self._run_tensorboard(output_dir)

        return

    def set_step_key(self, var_key):
        self._step_key = var_key
        return

    def log(self, key, val, collection=None, quiet=False):
        super().log(key, val, quiet)

        if (collection is not None):
            self._add_collection(collection, key)
        return

    def write_log(self):
        row_count = self._row_count

        super().write_log()

        if (self._writer is not None):
            if (row_count == 0):
                self._key_tags = self._build_key_tags()

            curr_time = time.time()
            step_val = row_count
            if (self._step_key is not None):
                step_val = self.log_current_row.get(self._step_key, "").val

            vals = []
            for i, key in enumerate(self.log_headers):
                if (key != self._step_key):
                    entry = self.log_current_row.get(key, "")
                    val = entry.val
                    tag = self._key_tags[i]
                    self._writer.add_scalar(tag, val, step_val)

        return
    
    def _add_collection(self, name, key):
        if (name not in self._collections):
            self._collections[name] = []
        self._collections[name].append(key)
        return

    def _delete_event_files(self, dir):
        if (os.path.exists(dir)):
            files = os.listdir(dir)
            for file in files:
                if ("events.out.tfevents." in file):
                    file_path = os.path.join(dir, file)
                    print("Deleting event file: {:s}".format(file_path))
                    os.remove(file_path)
        return

    def _build_key_tags(self):
        tags = []
        for key in self.log_headers:
            curr_tag = TBLogger.MISC_TAG
            for col_tag, col_keys in self._collections.items():
                if key in col_keys:
                    curr_tag = col_tag

            curr_tags = "{:s}/{:s}".format(curr_tag, key)
            tags.append(curr_tags)

        return tags

    def _run_tensorboard(self, output_dir):
        cmd = "tensorboard --logdir {:s}".format(output_dir)
        self._tb_proc = subprocess.Popen(cmd, shell=True)
        return