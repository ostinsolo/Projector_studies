# Two-plane physical setup required

## Status: USER_ACTION_REQUIRED

Complete these **physical** actions only. Do not capture images, edit JSON,
select points, or run commands — the agent will do all software steps after you
confirm readiness.

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
9. Do not move either device after confirming readiness.

Reply that the setup is ready. The agent will then run:

```
procam-calibrate auto-two-plane --run-dir <THIS_RUN_DIR> --mode physical --setup-confirmed
```
