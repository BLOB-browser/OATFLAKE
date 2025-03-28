# Import all the training_scheduler module
from .training_scheduler import start as training_scheduler_start
from .training_scheduler import stop as training_scheduler_stop
from .training_scheduler import get_status as training_scheduler_get_status
from .training_scheduler import set_training_time as training_scheduler_set_training_time

# Export training_scheduler functions under a namespace
training_scheduler = {
    'start': training_scheduler_start,
    'stop': training_scheduler_stop,
    'get_status': training_scheduler_get_status,
    'set_training_time': training_scheduler_set_training_time
}

# Don't try to import question_generator as it's used as a module with functions

__all__ = ['training_scheduler']
