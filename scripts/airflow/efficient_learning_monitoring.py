"""Same MBD/RPCBF computations, reusing exact training scores and batching validation."""
from full_learning_monitoring import *
import rmse_monitoring as original

def optimizer(env,cfg,basis,aware=True,wind_error_reserve=None,prepare_validation=False,information_trigger='stalled',allow_validation_slack=False):
    return original.optimizer(env,cfg,basis,aware,control_fn=control,reuse_training=True,batch_validation=True,wind_error_reserve=wind_error_reserve,prepare_validation=prepare_validation,information_trigger=information_trigger,allow_validation_slack=allow_validation_slack)
