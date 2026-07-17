# Two-Plane Physical Setup Request

## Status: SOFTWARE_READY_FOR_PHYSICAL_VALIDATION

Software (including architectural empty-scene analysis and extended synthetic
gates) is complete. Complete these **physical** actions only. Do not capture
images, edit JSON, select points, or run commands — the agent will execute all
software steps after you confirm readiness.

## User actions only

1. Use two rigid, matte, light-coloured flat panels or two walls meeting in a
   clear vertical corner.
2. Prefer an interior angle between approximately 60° and 100° for the first
   physical test.
3. Place the fold approximately near the centre third of the projector image,
   not directly at an extreme edge.
4. Ensure meaningful projected area exists on both planes.
5. Keep the projector fixed.
6. Keep the iPhone fixed.
7. Ensure the complete projection and the fold are visible in the iPhone frame
   (camera may be above, below, left, or right of the projector).
8. Darken the room enough for reliable ChArUco detection.
9. Disable macOS mirroring; use the external 1920×1080 projector display.
10. Do not move either device after confirming readiness.

## After confirmation

```bash
procam-calibrate auto-two-plane \
  --run-dir procam-test-data/runs/two_plane_phys_$(date +%Y%m%d_%H%M%S) \
  --mode physical \
  --setup-confirmed
```

See `docs/TWO_PLANE_PHYSICAL_RUNBOOK.md`.
