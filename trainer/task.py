from abc import ABCMeta, abstractmethod


class Task(metaclass=ABCMeta):
    @classmethod
    @abstractmethod
    def create(cls):
        return cls()

    @abstractmethod
    def start(self):
        raise NotImplementedError()

    @abstractmethod
    def tick(self):
        raise NotImplementedError()

    @abstractmethod
    def set_input(self, inputs):
        raise NotImplementedError()

    @abstractmethod
    def get_output(self):
        raise NotImplementedError()

    @abstractmethod
    def get_score(self):
        raise NotImplementedError()

    @abstractmethod
    def is_running(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_inputs(self):
        raise NotImplementedError()

    @property
    @abstractmethod
    def number_of_outputs(self):
        raise NotImplementedError()
