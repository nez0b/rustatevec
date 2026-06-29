// CUDA C kernels for the qsv GPU backend, compiled at runtime via NVRTC (cudarc).
//
// Layout mirrors the CPU backend exactly: Structure-of-Arrays, two f64 arrays `re`/`im`,
// index math identical to `crates/qsv-core/src/state/layout.rs`. Indices are 64-bit
// (`unsigned long long`) so n up to ~33 qubits fits. These are the *simplest correct* kernels
// (correctness-first, per the plan); coalescing / shared-memory / vectorized-load optimization
// comes after the differential suite is green.

typedef unsigned long long u64;

// --- bit-index helpers (device copies of state/layout.rs) ----------------------------------

// Insert a 0 bit at position `bit`, shifting higher bits up. Result is the |..0..> partner.
__device__ __forceinline__ u64 insert_zero_bit(u64 index, unsigned bit) {
    u64 mask = (1ULL << bit) - 1ULL;
    return (index & mask) | ((index & ~mask) << 1);
}

// Gather the bits of `index` at positions qs[0..m] into a compact sub-index.
__device__ __forceinline__ u64 gather_bits(u64 index, const int* qs, int m) {
    u64 sub = 0;
    for (int j = 0; j < m; ++j) sub |= ((index >> qs[j]) & 1ULL) << j;
    return sub;
}

// Scatter a compact sub-index back to full positions qs[0..m].
__device__ __forceinline__ u64 scatter_bits(u64 sub, const int* qs, int m) {
    u64 full = 0;
    for (int j = 0; j < m; ++j) full |= ((sub >> j) & 1ULL) << qs[j];
    return full;
}

// Insert a 0 bit at every position in `sorted` (strictly ascending). Produces the block base
// with all gate-qubit positions = 0.
__device__ __forceinline__ u64 insert_zero_bits(u64 index, const int* sorted, int m) {
    u64 r = index;
    for (int j = 0; j < m; ++j) r = insert_zero_bit(r, (unsigned)sorted[j]);
    return r;
}

// --- kernels -------------------------------------------------------------------------------

// Reset to a computational basis state |basis>: re[i] = (i == basis) ? 1 : 0, im[i] = 0.
extern "C" __global__ void k_init_basis(double* re, double* im, u64 basis, u64 dim) {
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    re[i] = (i == basis) ? 1.0 : 0.0;
    im[i] = 0.0;
}

// 1-qubit gate: thread t -> amplitude pair (a0, a1). g is the row-major 2x2 matrix as 8 scalars.
extern "C" __global__ void k_apply_1q(
    double* re, double* im, unsigned q,
    double g00r, double g00i, double g01r, double g01i,
    double g10r, double g10i, double g11r, double g11i,
    u64 n_pairs)
{
    u64 t = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= n_pairs) return;
    u64 a0 = insert_zero_bit(t, q);
    u64 a1 = a0 | (1ULL << q);

    double x0r = re[a0], x0i = im[a0];
    double x1r = re[a1], x1i = im[a1];

    re[a0] = g00r*x0r - g00i*x0i + g01r*x1r - g01i*x1i;
    im[a0] = g00r*x0i + g00i*x0r + g01r*x1i + g01i*x1r;
    re[a1] = g10r*x0r - g10i*x0i + g11r*x1r - g11i*x1i;
    im[a1] = g10r*x0i + g10i*x0r + g11r*x1i + g11i*x1r;
}

// Diagonal gate: psi_i *= diag[gather_bits(i, qubits)]. Fully coalesced (stride-1 over i).
extern "C" __global__ void k_apply_diagonal(
    double* re, double* im,
    const double* diagr, const double* diagi,
    const int* qubits, int m, u64 dim)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    u64 s = gather_bits(i, qubits, m);
    double dr = diagr[s], di = diagi[s];
    double xr = re[i], xi = im[i];
    re[i] = dr*xr - di*xi;
    im[i] = dr*xi + di*xr;
}

// General m-qubit gate (m <= 6): thread o -> one block of 2^m amplitudes. Gather 2^m sub-
// amplitudes, apply the dense matrix, scatter back. `qsorted` (ascending) drives the block
// base; `qorig` (gate convention order) drives gather/scatter. Matrix is row-major in global
// memory (mat[r*sub + c]). Shared-memory staging of the matrix is a later optimization.
extern "C" __global__ void k_apply_mq(
    double* re, double* im,
    const double* matr, const double* mati,
    const int* qsorted, const int* qorig, int m,
    u64 blocks)
{
    u64 o = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (o >= blocks) return;
    int sub = 1 << m;
    u64 base = insert_zero_bits(o, qsorted, m);

    double ar[64], ai[64];
    for (int s = 0; s < sub; ++s) {
        u64 idx = base | scatter_bits((u64)s, qorig, m);
        ar[s] = re[idx];
        ai[s] = im[idx];
    }
    for (int r = 0; r < sub; ++r) {
        double accr = 0.0, acci = 0.0;
        int row = r * sub;
        for (int c = 0; c < sub; ++c) {
            double mr = matr[row + c], mi = mati[row + c];
            accr += mr*ar[c] - mi*ai[c];
            acci += mr*ai[c] + mi*ar[c];
        }
        u64 idx = base | scatter_bits((u64)r, qorig, m);
        re[idx] = accr;
        im[idx] = acci;
    }
}

// Per-element squared magnitude: prob[i] = re[i]^2 + im[i]^2. (probabilities() reduction helper;
// the host sums or we add a reduction kernel later.)
extern "C" __global__ void k_abs2(
    const double* re, const double* im, double* out, u64 dim)
{
    u64 i = (u64)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= dim) return;
    double r = re[i], m = im[i];
    out[i] = r*r + m*m;
}
