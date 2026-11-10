"""Resolve one native EGL device to a CUDA UUID, without creating a GPU context."""
import argparse
import ctypes as C
import json
import os
import uuid


def audit(expected_uuid, egl_index=None):
    assert os.environ.get('CUDA_VISIBLE_DEVICES') == expected_uuid, 'UUID visibility required for audit'
    egl=C.CDLL('libEGL.so.1')
    egl.eglGetProcAddress.argtypes=[C.c_char_p]
    egl.eglGetProcAddress.restype=C.c_void_p
    query=C.CFUNCTYPE(C.c_uint,C.c_int,C.POINTER(C.c_void_p),C.POINTER(C.c_int))(
        egl.eglGetProcAddress(b'eglQueryDevicesEXT'))
    attribute=C.CFUNCTYPE(C.c_uint,C.c_void_p,C.c_int,C.POINTER(C.c_ssize_t))(
        egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
    devices=(C.c_void_p*64)();count=C.c_int()
    assert query(64,devices,C.byref(count)), 'EGL device enumeration failed'
    if egl_index is not None:
        assert 0 <= egl_index < count.value, 'EGL device index out of range'
    cuda=C.CDLL('libcuda.so.1')
    assert cuda.cuInit(0)==0, 'CUDA driver inventory failed'
    visible=C.c_int()
    assert cuda.cuDeviceGetCount(C.byref(visible))==0 and visible.value==1, 'expected exactly one visible CUDA device'
    candidates=range(count.value) if egl_index is None else [egl_index]
    for index in candidates:
        logical=C.c_ssize_t(-1)
        available=attribute(devices[index],0x323A,C.byref(logical))
        if not available:
            assert egl_index is None, 'EGL_CUDA_DEVICE_NV unavailable'
            continue
        value=(C.c_ubyte*16)()
        result=cuda.cuDeviceGetUuid(C.byref(value),int(logical.value))
        if result:
            assert egl_index is None, 'EGL CUDA device not visible'
            continue
        actual='GPU-'+str(uuid.UUID(bytes=bytes(value)))
        if actual!=expected_uuid:
            assert egl_index is None, ('EGL selected wrong GPU',actual,expected_uuid)
            continue
        return dict(egl_index=index,egl_device_count=count.value,cuda_logical_index=int(logical.value),
                    cuda_visible_devices=expected_uuid,actual_uuid=actual,context_created=False)
    raise AssertionError(('no EGL device resolves to expected UUID',expected_uuid))


if __name__=='__main__':
    ap=argparse.ArgumentParser()
    ap.add_argument('--uuid',required=True)
    ap.add_argument('--egl-index',type=int,default=None,
                    help='native EGL index; omit to discover the first device matching --uuid')
    args=ap.parse_args()
    print(json.dumps(audit(args.uuid,args.egl_index)))
