from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, HTMLResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from yoto.constants import DEFAULT_TAG_FAMILY
from yoto.tuning.yolo_dataset import builder as bd

_STATIC = Path(__file__).parent / "static"


def _load_cache(out_dir: Path) -> dict[str, dict]:
    cache: dict[str, dict] = {}
    for meta_path in sorted(out_dir.glob("*/meta.json")):
        df = pd.read_pickle(meta_path.parent / "candidates.pkl")
        cache[meta_path.parent.name] = {
            "meta": json.loads(meta_path.read_text()),
            "df": df,
            "dir": meta_path.parent,
        }
    return cache


def build_app(out_dir: Path) -> FastAPI:
    out_dir = Path(out_dir)
    cache = _load_cache(out_dir)
    session_path = out_dir / "session.json"

    first_thr = (
        next(iter(cache.values()))["meta"]["thresholds"]
        if cache
        else bd.Thresholds(
            area_min=100,
            area_max=1000,
            angle_min=70,
            angle_max=110,
            side_min=8,
            side_max=40,
            ratio_min=0.6,
            dedup_px=10.0,
        )
    )
    session = (
        bd.load_session(session_path)
        if session_path.exists()
        else bd.default_session(first_thr)  # type: ignore[arg-type]
    )
    # Threshold defaults always track the pkl-measured bounds from precompute.
    # A session.json left over from an earlier run (possibly a different pkl, or
    # sliders dragged to extremes) must not leave the sliders at wild values —
    # only the frame decisions and per-quad overrides in it are worth keeping.
    session["thresholds"] = dict(first_thr)

    # Lazily-built AprilTag detectors, one per tag family, shared by the live
    # "Enhance" re-detect path. Building one imports the apriltag fork and is
    # only paid the first time a reviewer ticks the box.
    detectors: dict[str, tuple[Any, dict]] = {}

    def _get_detector(family: str) -> tuple[Any, dict]:
        if family not in detectors:
            from yoto.detection import _build_apriltag_params_fast, _create_detector

            params = _build_apriltag_params_fast()
            detectors[family] = (_create_detector(params, family=family), params)
        return detectors[family]

    def _detect_candidates(exp: str, frame: int, enhance: bool) -> list[dict]:
        """Re-detect a frame from its saved JPG; returns candidate dicts.

        ``corners`` are 4x2 float arrays (as produced during precompute), so the
        result feeds both the JSON frame endpoint and ``run_export`` unchanged.
        """
        c = cache[exp]
        family = c["meta"].get("family", DEFAULT_TAG_FAMILY)
        detector, params = _get_detector(family)
        bgr = cv2.imread(str(c["dir"] / "images" / f"frame_{frame:06d}.jpg"))
        gray = cv2.cvtColor(bgr, cv2.COLOR_BGR2GRAY)
        valid, quads = bd.detect_full_frame(
            gray, {**params, "_family": family}, detector, enhance=enhance
        )
        valid_centers = (
            np.array([t["center"] for t in valid], dtype=np.float64)
            if valid
            else np.zeros((0, 2))
        )
        return bd.build_candidates_frame(quads, valid_centers)

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    def index() -> HTMLResponse:
        # Stamp the static files' newest mtime onto the JS/CSS URLs so the
        # browser refetches them whenever they change — otherwise an edited
        # app.js can stay cached while the new index.html loads, and handlers
        # silently go missing.
        html = (_STATIC / "index.html").read_text()
        ver = int(max(p.stat().st_mtime for p in _STATIC.glob("*.*")))
        html = html.replace("/static/app.js", f"/static/app.js?v={ver}")
        html = html.replace("/static/style.css", f"/static/style.css?v={ver}")
        return HTMLResponse(html)

    @app.get("/images/{exp}/{name}")
    def image(exp: str, name: str) -> FileResponse:
        return FileResponse(str(cache[exp]["dir"] / "images" / name))

    @app.get("/api/frames")
    def frames() -> list[dict]:
        out = []
        for exp, c in cache.items():
            for f in c["meta"]["frames"]:
                st = session["frames"].get(str(f), {}).get("status", "pending")
                out.append(
                    {
                        "exp": exp,
                        "frame": int(f),
                        "count": c["meta"]["frame_stats"].get(str(f), 0),
                        "status": st,
                    }
                )
        return out

    @app.get("/api/frame/{exp}/{frame}")
    def frame(exp: str, frame: int, enhance: bool = False) -> dict:
        c = cache[exp]
        if enhance:
            # Live re-detect this frame with the enhancement pre-stages on.
            cands = _detect_candidates(exp, frame, enhance=True)
            quads = [
                {
                    "quad_idx": int(r["quad_idx"]),
                    "corners": np.asarray(r["corners"]).reshape(4, 2).tolist(),
                    "is_valid": bool(r["is_valid"]),
                }
                for r in cands
            ]
        else:
            sub = c["df"][c["df"]["frame"] == frame]
            quads = [
                {
                    "quad_idx": int(r.quad_idx),
                    "corners": np.asarray(r.corners).reshape(4, 2).tolist(),
                    "is_valid": bool(r.is_valid),
                }
                for r in sub.itertuples()
            ]
        dec = session["frames"].get(str(frame), {})
        return {
            "image_url": f"/images/{exp}/frame_{frame:06d}.jpg",
            "width": c["meta"]["image_width"],
            "height": c["meta"]["image_height"],
            "quads": quads,
            "enhance": enhance,
            "overrides": dec.get("overrides", {}),
            "status": dec.get("status", "pending"),
        }

    @app.get("/api/thresholds")
    def get_thr() -> dict:
        return session["thresholds"]

    @app.get("/api/thresholds/default")
    def get_thr_default() -> dict:
        # The pkl-measured bounds from precompute — what "Reset to pkl bounds"
        # restores the sliders to.
        return dict(first_thr)

    @app.put("/api/thresholds")
    def put_thr(payload: dict) -> dict:
        session["thresholds"].update(payload)
        bd.save_session(session_path, session)
        return session["thresholds"]

    @app.put("/api/frame/{exp}/{frame}/decision")
    def decision(exp: str, frame: int, payload: dict) -> dict:
        overrides = {int(k): v for k, v in payload.get("overrides", {}).items()}
        bd.set_decision(
            session,
            frame,
            payload.get("status", "pending"),
            overrides,
            enhance=bool(payload.get("enhance", False)),
        )
        bd.save_session(session_path, session)
        return {"ok": True}

    @app.post("/api/accept-all")
    def accept_all() -> dict:
        n = 0
        for c in cache.values():
            for f in c["meta"]["frames"]:
                existing = session["frames"].get(str(f), {})
                overrides = {
                    int(k): v for k, v in existing.get("overrides", {}).items()
                }
                bd.set_decision(
                    session,
                    int(f),
                    "accepted",
                    overrides,
                    enhance=bool(existing.get("enhance", False)),
                )
                n += 1
        bd.save_session(session_path, session)
        return {"accepted": n}

    @app.post("/api/export")
    def export(payload: dict) -> JSONResponse:
        thr = bd.Thresholds(**session["thresholds"])  # type: ignore[typeddict-item]
        written = bd.run_export(
            cache,
            session,
            thr,
            out_dir,
            payload.get("formats", ["xanylabeling"]),
            float(payload.get("val_fraction", 0.2)),
            redetect=lambda e, f: _detect_candidates(e, f, enhance=True),
        )
        return JSONResponse({"written": written})

    return app


def run_server(out_dir: Path, host: str, port: int, open_browser: bool) -> None:
    import uvicorn

    print(f"Serving YOLO dataset builder at http://{host}:{port}")
    print(f"  Remote box? forward it:  ssh -L {port}:localhost:{port} <host>")
    if open_browser:
        import webbrowser

        webbrowser.open(f"http://{host}:{port}")
    uvicorn.run(build_app(out_dir), host=host, port=port)
