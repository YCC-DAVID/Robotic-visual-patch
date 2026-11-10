"""CPU max-min first-order fusion, invariant to positive loss rescaling.

This selects one feasible local update; it does not guarantee a nonlinear loss
increase or task failure. Native finite-step checks remain mandatory.
"""
import numpy as np
from scipy.optimize import linprog


def common_ascent(texture, gradients, step=.01):
    q=np.asarray(texture,dtype=np.float64)
    gs=np.asarray(gradients,dtype=np.float64)
    if gs.ndim!=q.ndim+1 or gs.shape[1:]!=q.shape:raise ValueError('gradient shape')
    if step<=0 or not np.isfinite(q).all() or not np.isfinite(gs).all():raise ValueError('non-finite inputs or step')
    if np.any(q<0) or np.any(q>1):raise ValueError('texture outside [0,1]')
    flat=gs.reshape(len(gs),-1)
    norms=np.abs(flat).sum(axis=1)
    active=np.any(flat!=0,axis=0)
    if not active.any() or np.any(norms==0):
        return dict(delta=np.zeros_like(q),minimum_gain=0.,normalized_gains=np.zeros(len(gs)),dual_weights=None,common_positive=False)
    # A positive loss-unit rescaling cancels exactly in g/||g||_1.
    unit=flat[:,active]/norms[:,None]
    qa=q.ravel()[active]
    bounds=list(zip(np.maximum(-1.,-qa/step),np.minimum(1.,(1-qa)/step)))+[(None,None)]
    costs=np.zeros(unit.shape[1]+1);costs[-1]=-1.
    constraints=np.column_stack([-unit,np.ones(len(gs))])
    result=linprog(costs,A_ub=constraints,b_ub=np.zeros(len(gs)),bounds=bounds,method='highs')
    if not result.success:raise RuntimeError(result.message)
    delta=np.zeros(q.size)
    positive=bool(result.x[-1]>1e-9)
    if positive:delta[active]=step*result.x[:-1]
    delta=delta.reshape(q.shape)
    gains=(flat@delta.ravel())/(step*norms)
    assert np.all(q+delta>=-1e-10) and np.all(q+delta<=1+1e-10)
    return dict(delta=delta,minimum_gain=float(np.min(gains)),normalized_gains=gains,
                dual_weights=-np.asarray(result.ineqlin.marginals),common_positive=positive)
