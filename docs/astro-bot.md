# Astro Bot (PPSA21564) on KytyPS5

This fork exists for one purpose: getting **Astro Bot** running on KytyPS5. Everything in the
`astro-bot-fixes` branch was found by running the game and fixing whatever stopped it next.

## Current state

- The game boots, plays the intro, reaches the title screen and the main menu, and a new save
  can be started.
- Rendering is complete (full frame, ray-traced lighting, UI).
- Still to do: a few materials render incorrectly, and the frame rate is not stable yet
  (roughly 17–55 fps in the intro on an RTX 3070 Ti / i7-11700K, the crowd and alien close-ups
  being the slowest; see the performance notes below).

## What was fixed, in the order it was found

1. **`IMAGE_BVH_INTERSECT_RAY` (MIMG 0xe6/0xe7)** — the game's lighting is ray traced and the
   shader recompiler rejected the opcode. Decoder, IR opcode (`BvhIntersectRay`), translator and a
   SPIR-V implementation were added (`spirvEmitterBvh.cpp`). The traversal is done in software:
   the BVH node is fetched through the BDA page table and box32 / box16 / triangle nodes are
   intersected the way the RDNA2 hardware does it (GPURT layout).
2. **Runtime-built buffer descriptors** — some shaders build their V# at runtime, so the SRT
   walker could not evaluate them. `S_BUFFER_LOAD` from a dynamic descriptor is lowered to a raw
   address load, and the host-side SRT evaluation treats unmapped memory as zero instead of
   faulting (a null TLAS pointer was crashing the emulator).
3. **Mip views past `max_mip`** — a compute mip-chain writer binds a view whose base level is
   beyond the descriptor's advertised max mip; the resource now extends to the view's last level.
4. **`VK_ERROR_DEVICE_LOST` in the per-pixel linked-list pass** — Vulkan helper invocations were
   running with EXEC active, so their (undefined) atomics corrupted the linked lists and the
   consumer looped forever. Pixel waves now start with `gl_HelperInvocation` lanes inactive, as on
   hardware.
5. **Structured control flow** — the CFG structurizer synthesizes a merge block when a selection's
   merge point is not inside the enclosing loop, so spirv-val accepts the ray-tracing shaders.
6. **`s_barrier` in vertex shaders** — emitted as a no-op outside compute/tessellation/mesh.
7. **SSE4a `EXTRQ` / `INSERTQ`** — the game uses them and the host CPU (Intel) does not have
   SSE4a; the register forms are emulated in the x64 instruction emulator.
8. **Indirect dispatch arguments written by the GPU** — the command processor reads
   `DISPATCH_INDIRECT` arguments from guest memory when it parses the packet, but the game's
   tile-lighting counts are produced by a compute pass in the same submission. When the argument
   triple lives in a cached GPU buffer, `vkCmdDispatchIndirect` is used instead of the CPU snapshot.
9. **Black right side of the screen / stuck sprites** — the depth target's HTile clear was applied
   as a render-pass load clear whose render area was limited by a stale 1024×1024 color target
   bound at the same time. Outside that area depth stayed 0, the tile classifier treated those
   tiles as sky, and the tile lighting never rewrote them, leaving old frame content behind.
   When the pass would not cover the whole depth image, the image is now cleared explicitly.

10. **Device memory exhaustion** — the game streams more textures than an 8 GB card holds;
    `vkCreateImage` failed fatally after a few minutes of the intro. Image creation now evicts
    least-recently-used textures and retries, and the garbage collector sweeps in bulk above the
    driver's reported budget.
11. **Per-frame GPU drains** — every CPU read of GPU-written memory submits and waits for the
    whole queue. The `DISPATCH_INDIRECT` snapshot read, the DCC clear-key fills (executed as
    compute, then read back on the next bind) and the game's own per-frame readbacks each cost a
    drain; those are now read without faulting, written on the host, and batched respectively.
    Non-mesh `DRAW_INDIRECT` is issued with `vkCmdDrawIndirect` instead of reading the arguments.
12. **Mesh-shader `DRAW_INDIRECT` on the GPU** — mesh programs read their draw data from a
    small device buffer (address in push constants); a one-thread compute shader converts the
    GPU-written arguments into that data plus a `VkDrawMeshTasksIndirectCommandEXT`. Guest
    threads that read GPU memory now wait for their own copy instead of stalling the command
    processor.
13. **Partially resident textures** — the game keeps the mip tails of streamed textures in
    128 KiB slots and points each T# base at where mip 0 would be, with `min_lod` marking the
    first resident mip. PS5 stores mip chains smallest-first, so the tail is a *prefix* of the
    allocation. The cache registered every such texture as a full 5.6 MiB image; the tails of ~40
    neighbouring slots overlapped each one, every tail the streamer wrote invalidated all of them,
    and each re-uploaded its whole chain (~800 MiB/s of uploads, 57 ms per frame in the crowd
    scene). Images now own only the byte span of their resident mips (registration, page
    tracking, invalidation, discovery and uploads), and a T# that lowers `min_lod` extends the
    residency and reloads the chain.
14. **Stale GPU-modified pages** — a page flagged GPU-modified with nothing pending to download
    never lost its flag on a read fault, so it stayed read-protected and the command processor
    faulted on the same shader header at every draw (77,000 page faults per second on the GPU
    thread). The flag is dropped when the download has nothing to fetch.
15. **Image rebinding** — resolving one image (or a colour target) can replace or expand another
    one bound earlier in the same draw; the rebind pass now repeats until every binding survives
    ("texture requires rediscovery before final acquisition").
16. **SRT walk cost** — memos are keyed on the user-data dwords the walk consumed (not every
    register), the evaluator's value table no longer spills to a hash map, and small guest reads
    hit a per-thread mapping cache instead of the address-space lock.
17. **Per-resource SRT memo** — the game allocates each draw's SRT struct from a ring, so
    memos keyed on user data never hit for the shaders that read their descriptor tables through
    that pointer. The evaluator now tracks which user-data dwords and which recorded reads every
    value depends on; reads addressed as "pointer + offset" (pointer from user data or from two
    earlier reads) are recorded relocatable and replayed at the current pointer, and only reads
    whose value reached a result have to match. Results that depend on memory alone (descriptor
    tables, flat constant slots, the control-flow decision) are copied into the walk when their
    reads still hold; per-draw descriptors are evaluated as before. Crowd scene ~20 → ~29 fps.
18. **Staging copies off the command processor** — uploads of 128 KiB and more are copied into
    the staging buffer by two worker threads; the scheduler joins them before each submit.
19. **Texture eviction ages** — the collector runs ~8 times per frame and its "pressured" tier
    is the normal state on an 8 GiB card; it evicted images unused for 10 frames (1 frame when
    over budget), which turned a tight VRAM budget into a re-upload storm. Ages are now 40 / 12 /
    4 frames.
20. **Readback race** — the asynchronous readback cleared a whole window's GPU-modified flag
    from the guest thread after the copy landed, losing writes the command processor had
    recorded meanwhile ("garbage collection retained GPU ownership"). Only pages without pending
    GPU bytes lose the flag, on the GPU thread.

## Performance notes (2026-09-17)

Measured with Tracy (`--profile`, `tracy-capture` / `tracy-csvexport`) during the intro
cutscene on the RTX 3070 Ti / i7-11700K:

- The emulator is **CPU-bound on the GPU thread**; the host GPU is not the limit. Making every
  ray miss (`KYTY_DEBUG_SKIP_BVH=1`) does not change the frame rate.
- Scripted 10-minute run (start, J×3, then hold W / press J in the desert): the intro's crowd
  scene went from 5 fps to ~29 fps, the bot close-up from 6 to ~30, the alien close-up from 3 to
  ~17–23; gameplay in the desert runs at 30–38 fps (idle 30–35, walking ~38) when VRAM stays
  under budget. The heavy scenes issue 600–1,100 draws and ~100–200 compute dispatches per frame.
- The loading tunnel after the intro still never finishes in roughly half of the runs
  ("infinite loading"). In that state the game's loader re-reads six normal maps forever and a
  single-thread finalize kernel (hash 0x9e3c6093e9c20738, `DS_APPEND` on four GDS counters, then
  dispatch arguments and a `-1` terminator per list) writes its 16-byte indirect arguments into
  the same page as a compute shader's code (0x50740a520 / 0x50740a700), so the command
  processor drains the queue once per frame reading that header. The lists are almost certainly
  the GPU-side resource requests the streamer consumes; what makes them loop is not known.
- What remains per frame in the crowd scene (Tracy self time): the SRT walk
  (`Srt::EvaluateRuntimeSources`, ~9 ms: ~880 walks of ~150 IR instructions and ~50 guest
  reads each; the memo hits only ~16% because the user data carries per-object constant-buffer
  V#s), texture re-uploads of textures the game re-streams (~4.5 ms), image create/delete churn
  from transient render-target aliasing (~2.5 ms), and a long tail of ~1 ms items.
- The game re-streams the same texture files continuously (~4 Hz per texture in gameplay; only
  a handful of normal maps during the loading tunnel). The cause is not understood; it also makes
  the loading tunnel occasionally never finish ("infinite loading"). `IMAGE_GET_LOD` is not used
  by the game, GPU page faults are zero, and the CPU readbacks are shader headers and constant
  data rather than a streaming feedback buffer.
- Blocking waits are reported by `KYTY_DEBUG_MEM_STATS=1` (`drains=` is the number of full
  queue drains, `gpu_faults=` the GPU-thread page faults with `KYTY_DEBUG_FAULT_TRACE=1`).

Two emulator instances on the same machine (for example running the game while a test run is
in progress) share VRAM and the GPU and make loading look stuck; test one at a time.

## Debugging aids that were kept

Environment variables (all off by default):

| Variable | Effect |
| --- | --- |
| `KYTY_DEBUG_SKIP_VIDEO=1` | `AvPlayerIsActive` reports every clip as finished (skips the intro videos). |
| `KYTY_DEBUG_SYNC_SUBMITS=1` | `vkQueueWaitIdle` after each submit and a `submit sync:` log line, so a device loss is attributed to the right command buffer. |
| `KYTY_DEBUG_LOG_DISPATCHES=1` | Log every compute dispatch (with shader hash and bound resources). |
| `KYTY_DEBUG_LOOP_LIMIT=N` | SPIR-V loop watchdog: a shader returns after N loop iterations in total (diagnostic only, breaks rendering). |
| `KYTY_DEBUG_DRAW_TARGET=<hex>` / `=1` | With `--graphics-debug-dump true`: log `DrawTargetState` for draws into that color target (or all draws). |
| `KYTY_DEBUG_DRAW_PS=<hex hash>` | Same, for draws using that pixel shader. |
| `KYTY_DEBUG_MEM_STATS=1` | Periodic line with device memory use, cache population, GPU-thread time split, blocking waits, drains, readbacks. |
| `KYTY_DEBUG_READBACK_TRACE=1` | Sampled log of the CPU reads that force a GPU drain (address, window). |
| `KYTY_DEBUG_SKIP_BVH=1` | Every ray query misses (profiling aid; lighting goes flat). |
| `KYTY_DEBUG_REFRESH_TRACE=1` | Log every guest re-upload of a registered image (address, size, format, why). |
| `KYTY_DEBUG_TEXTURE_TRACE=1` | Log the residency fields of mipmapped T#s and the residency decision with mip offsets. |
| `KYTY_DEBUG_FAULT_TRACE=1` | Count guest page faults (and GPU-thread faults) in the `KYTY_DEBUG_MEM_STATS` line. |
| `KYTY_DEBUG_READBACK_DUMP=<hex>` | Dump 64 dwords after a guest readback that faulted at that address. |
| `KYTY_DEBUG_SRT_MEMO=1` | Per-shader SRT memo statistics (hits, misses, walk time, prefills, read provenance). |
| `KYTY_DEBUG_SRT_DEPS=<hash,...>` | Dump a shader's user data and per-source dependency masks on its first walks. |
| `KYTY_DEBUG_SRT_RELOC=<hash>` | Dump a shader's recorded reads with their provenance and which read broke a prefill. |
| `KYTY_DEBUG_WATCH_ADDR=<hex>` / `KYTY_DEBUG_WATCH_SIZE=<hex>` | Log GPU write bindings, copies and fills that touch that guest range, with the shader. |
| `KYTY_NO_SRT_PREFILL=1`, `KYTY_NO_ASYNC_COPY=1` | Kill switches for the per-resource memo and the worker-thread staging copies. |
| `KYTY_DEBUG_IMAGE_TRACE=1` | Log image creation, replacement and expansion. |
| `KYTY_NO_RESIDENT_MIPS=1` | Treat every texture as fully resident (previous behaviour). |
| `KYTY_NO_TEXTURE_MEMO=1`, `KYTY_NO_SRT_MEMO=1`, `KYTY_NO_IMAGE_POOL=1`, `KYTY_NO_BARRIER_COALESCE=1`, `KYTY_NO_FLUSH_LIMIT=1`, `KYTY_NO_ASYNC_READBACK=1` | Kill switches for the individual optimizations. |

Useful CLI switches: `--shader-validation true` (spirv-val), `--shader-log-direction File`
(RDNA2 disassembly and IR in the printf log), `--graphics-debug-dump true --shader-log-folder DIR`
(shader `.bin` / `.rdna2` / `.spv` dumps).

Transient render-target addresses change between runs; trace by pixel-shader hash rather than by
address.

## Running

```
kyty_emulator.exe --game <path to PPSA21564-app>
```

Tested on Windows 10, RTX 3070 Ti, Intel i7-11700K.
