"""List native EGL devices (no GL context): index, extensions, DRM node, CUDA logical index/UUID. Read-only."""
import ctypes as C, os, socket, uuid, json
egl = C.CDLL('libEGL.so.1')
egl.eglGetProcAddress.argtypes = [C.c_char_p]; egl.eglGetProcAddress.restype = C.c_void_p
q = C.CFUNCTYPE(C.c_uint, C.c_int, C.POINTER(C.c_void_p), C.POINTER(C.c_int))(egl.eglGetProcAddress(b'eglQueryDevicesEXT'))
attr = C.CFUNCTYPE(C.c_uint, C.c_void_p, C.c_int, C.POINTER(C.c_ssize_t))(egl.eglGetProcAddress(b'eglQueryDeviceAttribEXT'))
qs = C.CFUNCTYPE(C.c_char_p, C.c_void_p, C.c_int)(egl.eglGetProcAddress(b'eglQueryDeviceStringEXT'))
devs = (C.c_void_p * 64)(); n = C.c_int(); assert q(64, devs, C.byref(n))
cuda = C.CDLL('libcuda.so.1'); cu_ok = cuda.cuInit(0) == 0
cnt = C.c_int(); cuda.cuDeviceGetCount(C.byref(cnt))
out = dict(host=socket.gethostname(), CUDA_VISIBLE_DEVICES=os.environ.get('CUDA_VISIBLE_DEVICES'), egl_count=n.value, cuda_count=cnt.value, devices=[])
for i in range(n.value):
    ext = (qs(devs[i], 0x3055) or b'').decode()           # EGL_EXTENSIONS
    drm = qs(devs[i], 0x3233) if 'EGL_EXT_device_drm' in ext else None   # EGL_DRM_DEVICE_FILE_EXT
    lg = C.c_ssize_t(-1); has = attr(devs[i], 0x323A, C.byref(lg))      # EGL_CUDA_DEVICE_NV
    u = None
    if has:
        v = (C.c_ubyte * 16)()
        if cuda.cuDeviceGetUuid(C.byref(v), int(lg.value)) == 0:
            u = 'GPU-' + str(uuid.UUID(bytes=bytes(v)))
    out['devices'].append(dict(egl=i, nvidia='EGL_NV_device_cuda' in ext, mesa='MESA' in ext, drm=drm.decode() if drm else None,
                               cuda_logical=int(lg.value) if has else None, uuid=u))
print(json.dumps(out))
