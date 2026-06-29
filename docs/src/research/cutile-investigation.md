# cuTile memory architecture — investigation

A deep-dive into NVIDIA's **cuTile** tile-based GPU programming model, written to answer one
question for this project: *should the GPU backend be written in cuTile (cuTile-rs), or in
hand-written CUDA C kernels driven from Rust (cudarc + NVRTC)?*

The short answer, derived below: **cuTile is real, mature enough to evaluate, and an excellent fit
for contiguous/GEMM-shaped work — but for our memory-bound, strided, complex-valued, non-GEMM
gate-apply kernel, hand-written cudarc/NVRTC kernels are the better fit**, because the workload's
performance hinges on exactly the low-level controls (coalescing, shared-memory staging, vectorized
complex loads, warp shuffles) that cuTile deliberately abstracts away. cuTile stays on the table as
a validation reference and for the contiguous (low-target-qubit) cases.

Sources: the two local clones under `_local/cutile-rs` and `_local/cutile-python`, the cuTile-Rust
paper *Fearless Concurrency on the GPU* (arXiv [2606.15991](https://arxiv.org/abs/2606.15991)),
NVIDIA's [cuTile-Python guide](https://docs.nvidia.com/cuda/cutile-python/), and the
[cuTile.jl blog](https://developer.nvidia.com/blog/cutile-jl-brings-nvidia-cuda-tile-based-programming-to-julia/).

## 1. Tiles and the memory hierarchy

cuTile splits state into two objects:

| Property | **Tensor** | **Tile** |
| --- | --- | --- |
| Location | Global memory (HBM) | Registers |
| Mutability | Mutable or read-only | Immutable |
| Shape | Static / dynamic / mixed | Static, powers of two |
| Operations | load / store only | arithmetic, reductions, matmul, shape ops |
| Lifetime | persists across kernels | exists only inside a kernel |
| Addressable | yes | no |

(`_local/cutile-rs/cutile-book/guide/tensors-and-tiles.md`; mirrored in the cuTile-Python guide.)

**A tensor is the addressable, strided object in HBM. A tile is a register-resident, statically
shaped fragment that exists only inside the kernel.** You load a tile from a tensor, compute on
tiles, and store a tile back.

### Who decides placement

The programmer chooses **tile shapes and access patterns**; the compiler owns **every level below
the tile**. From `useful-mental-models.md`:

> Registers — fastest storage; tiles live here during computation.
> Shared memory — fast on-chip storage shared within a hardware block.
> L2 cache — hardware-managed, shared across SMs.
> HBM — large global memory where tensors live.
>
> "In cuTile Rust, you load from tensors in HBM and compute on tiles in registers. The Tile IR
> compiler and runtime decide how to stage data through shared memory, caches, threads, warps,
> Tensor Cores, and Tensor Memory Accelerator (TMA) instructions when those mechanisms are useful."

So **shared-memory staging, TMA pipelining, and tensor-core (MMA) lowering are compiler-owned.** The
programmer's knobs are: tile shape, partition shape, dtype, algorithmic access pattern, an opt-in
`mma` call to reach tensor cores, and a few coarse `optimization_hints` (occupancy, CTA-in-CGA,
divisibility). The Triton-migration table in `interoperability.md` is explicit that *"the compiler
generates shared memory staging for `load_tile` operations"* and selects TMA automatically.

### Load / store / partition API

cuTile-Rust (`reference/dsl-api.md`):

```rust
tensor.load()                       // -> Tile          (load the output tile)
tensor.store(tile)                  //                  (store a tile to the tensor)
tensor.load_tile(shape, idx)        // -> Tile          (load at a partition index)
load_tile_like(src, dst)            // -> Tile          (load src at dst's tile-block position)

let part_x  = x.partition(const_shape![BM, BK]);   // device-side partition
let tile_x  = part_x.load([pid.0, k_tile]);
```

cuTile-Python (`samples/VectorAddition.py`):

```python
a_tile   = ct.load(a, index=(bid,), shape=(TILE,))   # HBM -> SMEM/registers (auto-distributed)
sum_tile = a_tile + b_tile
ct.store(c, index=(bid,), tile=sum_tile)             # tile -> HBM
```

The sample's own comment: *"`ct.load` automatically distributes the load across the threads within
the block, bringing the tile into shared memory or registers."* You do not write the
thread-index math.

## 2. Execution model vs SIMT

Traditional CUDA C++ is **thread-per-element**: you compute `i = blockIdx.x*blockDim.x + threadIdx.x`
and write `c[i] = a[i] + b[i]`. cuTile is **tile-per-block**: the entry function is written once as
straight-line, scalar-looking code over whole tiles, and the compiler performs the parallel
decomposition. From the paper:

> "The programmer writes sequential code over multi-dimensional tiles, and the compiler maps tile
> operations to thread blocks, manages shared memory, and performs the parallel decomposition."
>
> "Tile-based programming gives up SIMT-level control (explicit warp primitives, shared memory
> management) in exchange for the single-threaded semantics that make static safety checking
> tractable."

| Programmer controls | Automatic (compiler/runtime) |
| --- | --- |
| tile shape, partition shape, dtype | thread/warp assignment |
| algorithmic access pattern | **coalescing** |
| which ops (e.g. `mma` → tensor cores) | **shared-memory staging** |
| coarse `optimization_hints` | software pipelining (TMA), register allocation |

The critical caveat for us, from `useful-mental-models.md` (Coalescing and Strides): *"Tile loads
are designed to produce coalesced access patterns for regular layouts… Strided access can reduce
effective bandwidth because memory requests become scattered."* You cannot hand-fix coalescing — you
can only restructure the tile/partition shape and re-measure.

## 3. The Rust ownership / disjoint-partition model

cuTile-Rust extends Rust's aliasing-XOR-mutability across the launch boundary: **mutable output
tensors are partitioned into disjoint sub-tensors on the host before launch; immutable inputs are
shared.** The quick-start kernel (`_local/cutile-rs/README.md`):

```rust
#[cutile::entry()]
fn add<const B: i32>(
    z: &mut Tensor<f32, { [B] }>,   // exclusive mutable output
    x: &Tensor<f32, { [-1] }>,      // shared read-only
    y: &Tensor<f32, { [-1] }>,      // shared read-only
) {
    let tx = load_tile_like(x, z);
    let ty = load_tile_like(y, z);
    z.store(tx + ty);
}
// host:
let z = api::zeros::<f32>(&[1024]).partition([128]);  // disjoint 128-elt chunks
let (_z, _x, _y) = kernel::add(z, x, y).sync()?;      // grid (8,1,1) inferred
```

The invariant (paper): *"The mapping from tile programs to sub-tensors is injective: no sub-tensor
is assigned to more than one tile program."* The store target is the partition view itself, not an
index the programmer picks — so the classic index-swap data race is **inexpressible**. Memory
ordering is token-based: mutable references thread a `t₀ →load→ t₁ →store→ t₂` token chain
establishing happens-before; immutable references emit no tokens and may be freely reordered. Raw
`*mut T` device-pointer entries and `unchecked_accesses` are the opt-outs.

## 4. Host execution model

Every host-side call is a lazy `DeviceOp` (a GPU `Future`). Three modes over identical kernel code:

| Mode | API | Blocks? |
| --- | --- | --- |
| Synchronous | `.sync()` / `.sync_on(&stream)` | yes |
| Asynchronous | `.await` / `.schedule(policy)` | no (suspends task) |
| CUDA graph | `.graph()` / `.graph_on(stream)` | captures + replay |

Combinators mirror `futures` (`.then`, `zip!`, `.map`, `.shared`). CUDA-graph replay swaps inputs
and re-launches without recompilation (~0.8 µs/op overhead reported); only non-allocating ops
(kernel launches, `memcpy`) can be graph nodes. cuTile-Python is simpler:
`ct.launch(stream, grid, kernel, args)` plus a CUDA-graph path.

**Relevance to qsv:** this is genuinely attractive for a long gate sweep — capture the per-gate
launches once, replay across the circuit. Our `Backend::execute` override (batching the whole
circuit on one stream) maps cleanly onto either cuTile graphs or cudarc's manual graph API.

## 5. Requirements (and our box)

| | cuTile-Rust | cuTile-Python |
| --- | --- | --- |
| GPU | `sm_80`+ (Ada `sm_89` ✅) | Ampere/Ada/Blackwell (Hopper "coming") |
| CUDA toolkit | **13.2** for `sm_8x`, 13.3 recommended | **13.1+** |
| Driver | (toolkit-implied) | **r580+** |
| Rust | 1.89+ | — |
| OS | Linux (Ubuntu 24.04) | Linux |

**Our machine: 2× L40S (Ada `sm_89`, supported), but CUDA 12.4 / driver 550.127.08.** cuTile needs
**CUDA 13.2+ and driver r580+** — so it **cannot run here without a toolkit + driver upgrade**
(the driver bump is system-level / sysadmin). Note the irony: in early builds Ada is *better*
covered than Hopper (`sm_90` only landed in CUDA 13.3). Both projects self-describe as early-stage
("expect bugs, incomplete features, API breakage").

## 6. Suitability for our gate-apply kernel

Our kernel applies a 1- or 2-qubit gate to the statevector: an elementwise **complex** multiply
combined with a **strided** amplitude-pair gather/scatter (pairs separated by stride `2^t` for
target qubit `t`), **bandwidth-bound**, goal ~60–80% of HBM peak.

**cuTile is not GEMM-only.** Memory-bound elementwise is a first-class target: the paper reports
**7 TB/s for element-wise ops on B200 (~91% of peak HBM)**. So 60–80% of peak is a goal cuTile
routinely beats *for contiguous layouts*. It also *can* express gather/scatter: cuTile-Python ships
`ct.gather`/`ct.scatter` with index tiles; cuTile-Rust exposes `PointerTile`, `load_ptr_tko` /
`store_ptr_tko`, `addptr_tile`, and atomics. A clean framing is to **reshape** the statevector so
the target qubit splits the flat index into `[high, 2, low]` and make the size-2 pair axis a tile
dimension — then a low/contiguous-target gate becomes a coalesced tile load.

**But the real limitations for *this* workload:**

1. **No control over coalescing or SMEM staging — the crux.** A hand-written CUDA C kernel lets you
   choose the exact strided index math, use `double2`/`float4` vectorized loads, stage amplitude
   pairs in `__shared__`, use `__shfl_xor` for the in-warp pair exchange, and tune block size to the
   stride — the levers that take a strided gate kernel from ~40% to 70%+ of peak. cuTile removes
   those levers by design.
2. **Strided access is the one pattern cuTile explicitly flags as a risk.** `performance.md`:
   *"tile loads coalesce well, but algorithmic strides can still reduce effective bandwidth."* The
   bad pattern it names — `memory[0], memory[1024], memory[2048], …` — is exactly a
   **high-target-qubit** gate. The only recourse is to re-tile and re-measure, not hand-optimize.
3. **No native complex type.** The `ElementType` set is `f16/bf16/f32/f64`, ints, FP8/FP4, tf32 —
   **no complex**. You carry real/imag yourself (interleaved or a trailing size-2 axis) and write
   the complex multiply by hand — and the interleaved layout you'd want for coalescing is exactly
   what you can't force the compiler to honor.
4. **Maturity / portability gates:** early-stage, Linux-only, CUDA 13.2/13.3 + driver r580.

### cuTile vs hand-written cudarc/NVRTC

| Capability we need | cuTile | cudarc / NVRTC |
| --- | --- | --- |
| Explicit coalescing control | ✗ (compiler-decided) | ✅ |
| Shared-memory staging control | ✗ (auto) | ✅ (`__shared__`) |
| Vectorized loads (`double2`/`float4`) | ✗ | ✅ |
| Warp shuffles for pair exchange | ✗ (no warp primitives) | ✅ (`__shfl_xor`) |
| Exact strided index math | indirect (reshape/partition) | ✅ (direct) |
| Native complex layout | ✗ (manual interleave) | ✅ |
| Gather/scatter primitives | ✅ | ✅ |
| Race-free by construction | ✅ (ownership/partition) | ✗ (manual) |
| CUDA-graph replay | ✅ (`.graph()`) | ✅ (manual) |
| Maturity / portability | early, Linux, CUDA 13.2+ | mature, broad |
| Runs on our CUDA 12.4 box | ✗ | ✅ |

### Verdict

**For our memory-bound, strided, complex-valued, non-GEMM gate-apply kernel — targeting 60–80% of
HBM peak — hand-written cudarc/NVRTC kernels are the better fit.** Not because cuTile is GEMM-only
(it delivers ~91% of HBM peak on contiguous elementwise work), but because *this* workload's
performance hinges on exactly the controls cuTile abstracts away, its own docs name strided access
as the highest-risk pattern with no manual remedy, and it has no native complex type — and it can't
even run on this box's CUDA 12.4 today.

cuTile stays relevant as: (a) a fast-to-write **validation reference** once a CUDA-13 box is
available; (b) a strong option for the **contiguous / low-target-qubit** cases where a reshape makes
the gate axis a clean tile dimension; and (c) interop — cuTile-Rust's `borrow_raw` / `cudarc_interop`
lets a hand-tuned cudarc kernel handle the hard strided gates while cuTile handles the regular bulk,
sharing one `CUstream`. The structural race-freedom and graph-replay ergonomics are genuinely nice.
But to reliably hit the bandwidth roof on the strided cases, we need the low-level memory control
that only a hand-written kernel provides — so **qsv's `CudaBackend` is built on cudarc + NVRTC**
(see the [GPU landscape note](gpu-and-rust.md)).
