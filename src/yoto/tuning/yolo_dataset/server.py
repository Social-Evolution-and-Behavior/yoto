from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from fastapi import FastAPI
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

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

    app = FastAPI()
    app.mount("/static", StaticFiles(directory=str(_STATIC)), name="static")

    @app.get("/")
    def index() -> FileResponse:
        return FileResponse(str(_STATIC / "index.html"))

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
    def frame(exp: str, frame: int) -> dict:
        c = cache[exp]
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
            "overrides": dec.get("overrides", {}),
            "status": dec.get("status", "pending"),
        }

    @app.get("/api/thresholds")
    def get_thr() -> dict:
        return session["thresholds"]

    @app.put("/api/thresholds")
    def put_thr(payload: dict) -> dict:
        session["thresholds"].update(payload)
        bd.save_session(session_path, session)
        return session["thresholds"]

    @app.put("/api/frame/{exp}/{frame}/decision")
    def decision(exp: str, frame: int, payload: dict) -> dict:
        overrides = {int(k): v for k, v in payload.get("overrides", {}).items()}
        bd.set_decision(session, frame, payload.get("status", "pending"), overrides)
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
                bd.set_decision(session, int(f), "accepted", overrides)
                n += 1
        bd.save_session(session_path, session)
        return {"accepted": n}

    @app.post("/api/export")
    def export(payload: dict) -> JSONResponse:
        thr = bd.Thresholds(**session["thresholds"])  # type: ignore[arg-type]
        written = bd.run_export(
            cache,
            session,
            thr,
            out_dir,
            payload.get("formats", ["labelme"]),
            float(payload.get("val_fraction", 0.2)),
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
