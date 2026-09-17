# Astro Bot (PPSA21564) on KytyPS5

This fork exists for one purpose: getting **Astro Bot** running on KytyPS5. Everything in the
`astro-bot-fixes` branch was found by running the game and fixing whatever stopped it next.

## Current state

- The game boots, plays the intro, reaches the title screen and the main menu, and a new save
  can be started.
- Rendering is complete (full frame, ray-traced lighting, UI).
- Still to do: a few materials render incorrectly, and the frame rate is low (roughly 25–30 fps
  on an RTX 3070 Ti / i7-11700K at the game's dynamic resolution).

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
