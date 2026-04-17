"""Basic YOTO pipeline example.

Demonstrates the three-step workflow: detect, clean, render.

Usage:
    python examples/basic_pipeline.py path/to/video.mp4
"""

from __future__ import annotations

import sys

from yoto import clean_tracking_data, render_overlay_video, run_detection_simple


def main() -> None:
    """Run the basic pipeline on a single video."""
    if len(sys.argv) < 2:
        print("Usage: python examples/basic_pipeline.py <video_path>")
        sys.exit(1)

    video_path = sys.argv[1]

    # Step 1: Detect AprilTags using the simple (portable) pipeline
    print("Step 1: Running detection...")
    df = run_detection_simple(video_path, yolo_weights="detect14.engine")

    # Step 2: Clean and interpolate the raw tracking data
    print("Step 2: Cleaning data...")
    cleaned, ids, metrics = clean_tracking_data(df)
    print(
        f"  Detected {len(ids)} tags | "
        f"Error rate: {metrics['error_pct']:.2f}% | "
        f"Gaps filled: {metrics['filled_pct_of_gaps']:.1f}%"
    )

    # Step 3: Render overlay video
    print("Step 3: Rendering overlay video...")
    output = render_overlay_video(video_path, cleaned, ids, scale=0.5)
    print(f"  Output: {output}")


if __name__ == "__main__":
    main()
