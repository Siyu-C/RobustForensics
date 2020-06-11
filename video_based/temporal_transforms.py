import random


class TemporalCenterCrop(object):
    """Temporally crop the given frame indices at a center.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    
    Sample the resulting frames with a fixed step size.

    Args:
        size (int): Desired temporal length of the crop.
        step (int): Temporal sampling frequency.
        adjacent (int): Group adjacent frames together when sampling.
    """

    def __init__(self, size, step=1, adjacent=1):
        self.size = size
        self.step = step
        self.adjacent = adjacent

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """

        center_index = len(frame_indices) // 2
        begin_index = max(0, center_index - (self.size // 2))
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        sampled_out = []
        for index in range(0, self.size - self.adjacent + 1, self.step * self.adjacent):
            for delta in range(self.adjacent):
                sampled_out.append(out[index + delta])
        return sampled_out


class TemporalRandomCrop(object):
    """Temporally crop the given frame indices at a random location.

    If the number of frames is less than the size,
    loop the indices as many times as necessary to satisfy the size.
    
    Sample the resulting frames with a fixed step size.

    Args:
        size (int): Desired temporal length of the crop.
        step (int): Temporal sampling frequency.
        adjacent (int): Group adjacent frames together when sampling.
    """

    def __init__(self, size, step=1, adjacent=1):
        self.size = size
        self.step = step
        self.adjacent = adjacent

    def __call__(self, frame_indices):
        """
        Args:
            frame_indices (list): frame indices to be cropped.
        Returns:
            list: Cropped frame indices.
        """
        
        rand_end = max(0, len(frame_indices) - self.size - 1)
        begin_index = random.randint(0, rand_end)
        end_index = min(begin_index + self.size, len(frame_indices))

        out = frame_indices[begin_index:end_index]

        for index in out:
            if len(out) >= self.size:
                break
            out.append(index)

        sampled_out = []
        for index in range(0, self.size - self.adjacent + 1, self.step * self.adjacent):
            for delta in range(self.adjacent):
                sampled_out.append(out[index + delta])
        return sampled_out


class TemporalSampling(object):

    def __init__(self, step=1, adjacent=1):
        self.step = step
        self.adjacent = adjacent

    def __call__(self, frame_indices):
        sampled_out = []
        for index in range(0, len(frame_indices) - self.adjacent + 1, self.step * self.adjacent):
            for delta in range(self.adjacent):
                sampled_out.append(frame_indices[index + delta])
        return sampled_out
