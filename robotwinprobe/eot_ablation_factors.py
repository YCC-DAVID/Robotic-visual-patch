"""Paired only-one / leave-one-out masks over the exact existing EOT draws."""
from texture_eot import IDENTITY


FACTORS={
    'rotation':('angle_deg',),
    'translation':('dx_fraction','dy_fraction'),
    'scale':('scale',),
    'gain':('gain',),
}
VARIANTS={**{'only_'+k:[k] for k in FACTORS},
          **{'drop_'+k:[x for x in FACTORS if x!=k] for k in FACTORS}}
LABELS=dict(rotation='旋转',translation='二维平移',scale='缩放',gain='RGB统一增益')


def mask_draw(draw,active):
    unknown=set(active)-FACTORS.keys()
    if unknown:raise ValueError('Unknown EOT factors: '+str(unknown))
    masked=dict(IDENTITY)
    for factor in active:
        for key in FACTORS[factor]:masked[key]=draw[key]
    return masked


def masked_plan(draws,variant):
    return [[mask_draw(draw,VARIANTS[variant]) for draw in batch] for batch in draws]
