import torch

class ExperienceBuffer():
    def __init__(self, buffer_length, device):
        self._buffer_length = buffer_length
        self._device = device

        self._buffer_head = 0
        self._total_samples = 0

        self._buffers = dict()
        self._sample_buf = torch.randperm(self._buffer_length, device=self._device,
                                          dtype=torch.long)
        self._sample_buf_head = 0
        self._reset_sample_buf()

        return

    def add_buffer(self, name, buffer):
        assert(len(buffer.shape) >= 1)
        assert(buffer.shape[0] == self._buffer_length)
        assert(name not in self._buffers)

        self._buffers[name] = buffer

        return

    def reset(self):
        self._buffer_head = 0
        self._reset_sample_buf()
        return

    def clear(self):
        self.reset()
        self._total_samples = 0
        return

    def inc(self):
        self._buffer_head = (self._buffer_head + 1) % self._buffer_length
        self._total_samples += 1
        return

    def get_total_samples(self):
        return self._total_samples

    def get_sample_count(self):
        sample_count = min(self._total_samples, self._buffer_length)
        return sample_count

    def record(self, name, data):
        data_buf = self._buffers[name]
        data_buf[self._buffer_head] = data
        return

    def set_data(self, name, data):
        data_buf = self._buffers[name]
        assert(data_buf.shape[0] == data.shape[0])
        data_buf[:] = data
        return
    
    def get_data(self, name):
        return self._buffers[name]

    def sample(self, n):
        output = dict()
        rand_idx = self._sample_rand_idx(n)

        for key, data in self._buffers.items():
            batch_data = data[rand_idx]
            output[key] = batch_data

        return output

    def _reset_sample_buf(self):
        self._sample_buf[:] = torch.randperm(self._buffer_length, device=self._device, dtype=torch.long)
        self._sample_buf_head = 0
        return

    def _sample_rand_idx(self, n):
        buffer_len = self._sample_buf.shape[0]
        assert(n <= buffer_len)

        if (self._sample_buf_head + n <= buffer_len):
            rand_idx = self._sample_buf[self._sample_buf_head:self._sample_buf_head + n]
            self._sample_buf_head += n
        else:
            rand_idx0 = self._sample_buf[self._sample_buf_head:]
            remainder = n - (buffer_len - self._sample_buf_head)

            self._reset_sample_buf()
            rand_idx1 = self._sample_buf[:remainder]
            rand_idx = torch.cat([rand_idx0, rand_idx1], dim=0)

            self._sample_buf_head = remainder

        sample_count = self.get_sample_count()
        rand_idx = torch.remainder(rand_idx, sample_count)
        return rand_idx