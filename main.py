import os
import uuid
import threading
import time
from flask import Flask, render_template, request, jsonify, send_file, abort
from screen_extractor import process_video

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 500 * 1024 * 1024  # 500 MB

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), "outputs")
os.makedirs(UPLOAD_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# job_id -> {"status": "pending"|"processing"|"done"|"error", "progress": 0-100, "message": str}
jobs: dict[str, dict] = {}
jobs_lock = threading.Lock()


def _run_job(job_id: str, input_path: str, output_path: str):
    def progress(done, total):
        pct = int(done / total * 100) if total else 0
        with jobs_lock:
            jobs[job_id]["progress"] = pct

    with jobs_lock:
        jobs[job_id]["status"] = "processing"

    try:
        info = process_video(input_path, output_path, progress_callback=progress)
        with jobs_lock:
            jobs[job_id].update({"status": "done", "progress": 100,
                                  "info": info, "output_path": output_path})
    except Exception as e:
        with jobs_lock:
            jobs[job_id].update({"status": "error", "message": str(e)})
    finally:
        try:
            os.remove(input_path)
        except OSError:
            pass


@app.route("/")
def index():
    return render_template("index.html")


@app.route("/upload", methods=["POST"])
def upload():
    if "video" not in request.files:
        return jsonify({"error": "No file provided"}), 400
    f = request.files["video"]
    if not f.filename:
        return jsonify({"error": "Empty filename"}), 400

    ext = os.path.splitext(f.filename)[1].lower()
    if ext not in (".mp4", ".mov", ".avi", ".mkv", ".m4v", ".3gp"):
        return jsonify({"error": "Unsupported format. Use MP4, MOV, AVI, MKV."}), 400

    job_id = str(uuid.uuid4())
    input_path = os.path.join(UPLOAD_DIR, f"{job_id}{ext}")
    output_path = os.path.join(OUTPUT_DIR, f"{job_id}_extracted.mp4")

    f.save(input_path)

    with jobs_lock:
        jobs[job_id] = {"status": "pending", "progress": 0, "message": ""}

    t = threading.Thread(target=_run_job, args=(job_id, input_path, output_path), daemon=True)
    t.start()

    return jsonify({"job_id": job_id})


@app.route("/status/<job_id>")
def status(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None:
        abort(404)
    return jsonify(job)


@app.route("/download/<job_id>")
def download(job_id: str):
    with jobs_lock:
        job = jobs.get(job_id)
    if job is None or job.get("status") != "done":
        abort(404)
    path = job.get("output_path", "")
    if not os.path.exists(path):
        abort(404)
    return send_file(path, mimetype="video/mp4",
                     as_attachment=True, download_name="extracted_screen.mp4")


if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False)
