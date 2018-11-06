from abc import ABCMeta, abstractmethod


class Task(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def create(cls):
        # Should return an instance of the task.
        raise NotImplementedError()

    @abstractmethod
    def start(self):
        # Should start the task, must be repeatable.
        raise NotImplementedError()

    @abstractmethod
    def tick(self):
        # Should bring the task forward. Can be connected to
        # game loop or just a sleep.
        raise NotImplementedError()

    @abstractmethod
    def set_input(self, inputs):
        # Inputs are returned as a value in the interval [0, 1],
        # if a binary input is required it can for example be
        # implemented as inputs[n] > 0.5.
        raise NotImplementedError()

    @abstractmethod
    def get_output(self):
        # Should return the output state, this must be normalised
        # to a value in the interval ]0, 1[.
        raise NotImplementedError()

    @abstractmethod
    def get_score(self):
        # Should return a score in the interval ]0, 1000[ so that the
        # neural trainer can evaluate how well an 'organism' has
        # performed. Called after the task has finished.
        raise NotImplementedError()

    @abstractmethod
    def is_running(self):
        # Called after each tick, to see if it should continue.
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_inputs(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_outputs(self):
        raise NotImplementedError()
