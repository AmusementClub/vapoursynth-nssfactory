/* Linux/glibc diagnostic only: cc -shared -fPIC -O2 c4_alloc_probe.c -o alloc.so
 * LD_PRELOAD=alloc.so python c4_bm_memory.py ...
 * Counts allocation requests process-wide, including Python and VapourSynth.
 * It deliberately does not claim these are NSS-only allocations or live bytes.
 */
#include <errno.h>
#include <stddef.h>
#include <stdint.h>
#include <stdatomic.h>
extern void *__libc_malloc(size_t);
extern void *__libc_calloc(size_t, size_t);
extern void *__libc_realloc(void *, size_t);
extern void __libc_free(void *);
extern void *__libc_memalign(size_t, size_t);
static _Atomic uint64_t requests, bytes;
static void count(size_t n) {
    atomic_fetch_add_explicit(&requests, 1, memory_order_relaxed);
    atomic_fetch_add_explicit(&bytes, n, memory_order_relaxed);
}
uint64_t nss_probe_requests(void) { return atomic_load_explicit(&requests, memory_order_relaxed); }
uint64_t nss_probe_bytes(void) { return atomic_load_explicit(&bytes, memory_order_relaxed); }
void *malloc(size_t n) { count(n); return __libc_malloc(n); }
void *calloc(size_t n, size_t s) { count(n*s); return __libc_calloc(n,s); }
void *realloc(void *p, size_t n) { count(n); return __libc_realloc(p,n); }
void free(void *p) { __libc_free(p); }
void *aligned_alloc(size_t a, size_t n) { count(n); return __libc_memalign(a,n); }
void *memalign(size_t a, size_t n) { count(n); return __libc_memalign(a,n); }
int posix_memalign(void **p, size_t a, size_t n) {
    if (a < sizeof(void*) || (a & (a-1))) return EINVAL;
    count(n);
    void *q = __libc_memalign(a,n);
    if (!q) return ENOMEM;
    *p=q;
    return 0;
}
