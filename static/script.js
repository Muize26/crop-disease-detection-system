/* ────────────────────────────────────────────────────────────────
                                                                                                                                                                                                       STATE VARIABLES
                                                                                                                                                                                                       These track what input mode we're in and what image is ready.
                                                                                                                                                                                                    ──────────────────────────────────────────────────────────────── */
let activeTab = "upload"; // 'upload' or 'camera'
let cameraStream = null; // MediaStream object when camera is active
let capturedImageB64 = null; // Base64 string from camera snapshot
let reportImageData = null; // Base64 string used for PDF reports

/* ────────────────────────────────────────────────────────────────
     TAB SWITCHING (Upload ↔ Camera)
  ──────────────────────────────────────────────────────────────── */
function switchTab(tab) {
    activeTab = tab;

    // Update tab button styles
    document
        .getElementById("tab-upload")
        .classList.toggle("active", tab === "upload");
    document
        .getElementById("tab-camera")
        .classList.toggle("active", tab === "camera");

    // Show / hide panels
    document.getElementById("upload-panel").style.display =
        tab === "upload" ? "block" : "none";
    document.getElementById("camera-panel").style.display =
        tab === "camera" ? "block" : "none";

    // Stop camera if switching away
    if (tab === "upload") stopCamera();
}

/* ────────────────────────────────────────────────────────────────
     FILE UPLOAD — preview thumbnail
  ──────────────────────────────────────────────────────────────── */
const fileInput = document.getElementById("file-input");
const previewImg = document.getElementById("preview-img");
const dropZone = document.getElementById("drop-zone");

fileInput.addEventListener("change", () => {
    const file = fileInput.files[0];
    if (!file) return;
    const url = URL.createObjectURL(file);
    previewImg.src = url;
    previewImg.style.display = "block";

    const reader = new FileReader();
    reader.onload = () => {
        reportImageData = reader.result;
    };
    reader.readAsDataURL(file);
});

// Drag-and-drop visual feedback
dropZone.addEventListener("dragover", (e) => {
    e.preventDefault();
    dropZone.classList.add("drag-over");
});
dropZone.addEventListener("dragleave", () =>
    dropZone.classList.remove("drag-over"),
);
dropZone.addEventListener("drop", (e) => {
    e.preventDefault();
    dropZone.classList.remove("drag-over");
    if (e.dataTransfer.files.length) {
        fileInput.files = e.dataTransfer.files;
        fileInput.dispatchEvent(new Event("change"));
    }
});

/* ────────────────────────────────────────────────────────────────
     CAMERA — start, capture, retake, stop
  ──────────────────────────────────────────────────────────────── */
async function startCamera() {
    const video = document.getElementById("camera-video");
    const status = document.getElementById("camera-status");
    const btnStart = document.getElementById("btn-start-camera");
    const btnCap = document.getElementById("btn-capture");

    status.textContent = "Requesting camera permission…";

    try {
        // Ask the browser for webcam access.
        // On HTTPS or localhost this shows a permission popup.
        cameraStream = await navigator.mediaDevices.getUserMedia({
            video: {
                facingMode: "environment",
                width: {
                    ideal: 1280,
                },
                height: {
                    ideal: 960,
                },
            },
            audio: false,
        });

        video.srcObject = cameraStream;
        await video.play();

        status.textContent = "Camera active — point at a leaf and capture.";
        btnStart.style.display = "none";
        btnCap.style.display = "flex";
        capturedImageB64 = null;
        document.getElementById("camera-snapshot").style.display = "none";
    } catch (err) {
        // Common errors:
        // NotAllowedError  — user denied permission
        // NotFoundError    — no camera device found
        status.textContent = `⚠ Camera error: ${err.name}. Check permissions.`;
        console.error("Camera error:", err);
    }
}

function capturePhoto() {
    const video = document.getElementById("camera-video");
    const canvas = document.getElementById("camera-canvas");
    const snapshot = document.getElementById("camera-snapshot");
    const btnCap = document.getElementById("btn-capture");
    const btnRet = document.getElementById("btn-retake");
    const status = document.getElementById("camera-status");

    // Draw the current video frame onto a hidden <canvas>
    canvas.width = video.videoWidth || 640;
    canvas.height = video.videoHeight || 480;
    const ctx = canvas.getContext("2d");
    ctx.drawImage(video, 0, 0, canvas.width, canvas.height);

    // Convert canvas to a Base64 PNG string
    capturedImageB64 = canvas.toDataURL("image/png");
    reportImageData = capturedImageB64;

    // Show the snapshot preview
    snapshot.src = capturedImageB64;
    snapshot.style.display = "block";

    // Pause / hide live video
    video.pause();

    btnCap.style.display = "none";
    btnRet.style.display = "flex";
    status.textContent = "✅ Photo captured! Click Predict Disease.";
}

function retakePhoto() {
    const video = document.getElementById("camera-video");
    const snapshot = document.getElementById("camera-snapshot");
    const btnCap = document.getElementById("btn-capture");
    const btnRet = document.getElementById("btn-retake");
    const status = document.getElementById("camera-status");

    capturedImageB64 = null;
    snapshot.style.display = "none";
    video.play();

    btnRet.style.display = "none";
    btnCap.style.display = "flex";
    status.textContent = "Camera active — point at a leaf and capture.";
}

function stopCamera() {
    if (cameraStream) {
        cameraStream.getTracks().forEach((t) => t.stop());
        cameraStream = null;
    }
    const video = document.getElementById("camera-video");
    const snapshot = document.getElementById("camera-snapshot");
    const btnStart = document.getElementById("btn-start-camera");
    const btnCap = document.getElementById("btn-capture");
    const btnRet = document.getElementById("btn-retake");
    const status = document.getElementById("camera-status");

    video.srcObject = null;
    snapshot.style.display = "none";
    capturedImageB64 = null;

    btnStart.style.display = "flex";
    btnCap.style.display = "none";
    btnRet.style.display = "none";
    status.textContent = 'Click "Start Camera" to begin.';
}

/* ────────────────────────────────────────────────────────────────
     MAIN PREDICTION FUNCTION
     Called when the user clicks "Predict Disease".
     Decides whether to POST a file (upload mode) or Base64 (camera mode).
  ──────────────────────────────────────────────────────────────── */
async function runPrediction() {
    const cropType = document.getElementById("crop-select").value;
    const btn = document.getElementById("predict-btn");
    const label = document.getElementById("btn-label");
    const spinner = document.getElementById("btn-spinner");

    // Show loading state
    btn.disabled = true;
    label.textContent = "Analysing…";
    spinner.style.display = "block";

    try {
        let result;

        if (activeTab === "upload") {
            // ── Upload mode: send image as multipart/form-data ────────────
            const file = fileInput.files[0];
            if (!file) {
                showError("Please upload an image first.");
                resetPredictButton(btn, label, spinner);
                return;
            }
            reportImageData = reportImageData || "";

            const formData = new FormData();
            formData.append("crop", cropType);
            formData.append("image", file);

            const response = await fetch("/predict", {
                method: "POST",
                body: formData, // No Content-Type header — browser sets it automatically
            });
            result = await response.json();
        } else {
            // ── Camera mode: send Base64 image as JSON ────────────────────
            if (!capturedImageB64) {
                showError("Please capture a photo first.");
                resetPredictButton(btn, label, spinner);
                return;
            }

            const response = await fetch("/predict_camera", {
                method: "POST",
                headers: {
                    "Content-Type": "application/json",
                },
                body: JSON.stringify({
                    crop: cropType,
                    image: capturedImageB64,
                }),
            });
            result = await response.json();
        }

        // ── Handle error from Flask ────────────────────────────────────
        if (result.error) {
            showError(result.error);
            resetPredictButton(btn, label, spinner);
            return;
        }

        // ── Save report metadata and display result ─────────────────────
        window.latestReportPayload = {
            crop: cropType,
            disease: result.disease,
            confidence: result.confidence,
            suggestion: result.suggestion,
            image_data: reportImageData || "",
        };

        displayResult(result);
    } catch (err) {
        showError("Network error — is Flask running? " + err.message);
        resetPredictButton(btn, label, spinner);
    }
}

function resetPredictButton(btn, label, spinner) {
    btn.disabled = false;
    label.textContent = "Predict Disease";
    spinner.style.display = "none";
}

/* ────────────────────────────────────────────────────────────────
     DISPLAY RESULT in the right card
  ──────────────────────────────────────────────────────────────── */
function displayResult(data) {
    const { disease, confidence, suggestion } = data;

    // Hide placeholder, show results
    document.getElementById("result-placeholder").style.display = "none";
    document.getElementById("result-content").style.display = "block";

    // Disease badge — colour based on result
    const badge = document.getElementById("disease-badge");
    badge.textContent = disease;
    badge.className = "disease-badge";
    if (disease.toLowerCase().includes("healthy")) {
        // keep default green
    } else if (
        disease.toLowerCase().includes("blight") ||
        disease.toLowerCase().includes("mosaic")
    ) {
        badge.classList.add("danger");
    } else {
        badge.classList.add("alert");
    }

    // Confidence bar
    document.getElementById("conf-pct").textContent = confidence + "%";
    // Small delay so the CSS transition is visible
    setTimeout(() => {
        document.getElementById("conf-bar").style.width = confidence + "%";
    }, 50);

    // Suggestion
    document.getElementById("suggestion-text").textContent = suggestion;

    // Scroll result into view on mobile
    document.getElementById("result-panel").scrollIntoView({
        behavior: "smooth",
        block: "nearest",
    });

    // Change button to Clear after successful prediction
    const btn = document.getElementById("predict-btn");
    const label = document.getElementById("btn-label");
    const downloadBtn = document.getElementById("download-report-btn");
    btn.disabled = false;
    label.textContent = "Clear";
    btn.onclick = clearResults;
    downloadBtn.style.display = "flex";
}

/* ────────────────────────────────────────────────────────────────
     CLEAR / RESET FUNCTION
     Resets all data and UI to initial state without page reload
  ──────────────────────────────────────────────────────────────── */
function clearResults() {
    // ── Hide results, show placeholder ─────────────────────────────────
    document.getElementById("result-content").style.display = "none";
    document.getElementById("result-placeholder").style.display = "block";

    // ── Reset confidence bar ───────────────────────────────────────────
    document.getElementById("conf-pct").textContent = "0%";
    document.getElementById("conf-bar").style.width = "0%";

    // ── Clear disease and suggestion text ──────────────────────────────
    document.getElementById("disease-badge").textContent = "—";
    document.getElementById("disease-badge").className = "disease-badge";
    document.getElementById("suggestion-text").textContent = "—";

    // ── Reset upload input ─────────────────────────────────────────────
    fileInput.value = "";
    previewImg.style.display = "none";
    previewImg.src = "";
    reportImageData = null;

    // ── Reset camera ──────────────────────────────────────────────────
    stopCamera();
    document.getElementById("camera-snapshot").style.display = "none";
    capturedImageB64 = null;

    // ── Reset button to "Predict Disease" ──────────────────────────────
    const btn = document.getElementById("predict-btn");
    const label = document.getElementById("btn-label");
    const spinner = document.getElementById("btn-spinner");
    label.textContent = "Predict Disease";
    spinner.style.display = "none";
    btn.disabled = false;
    btn.onclick = runPrediction;

    // ── Optional: scroll to analysis section ────────────────────────────
    document.getElementById("analysis-section").scrollIntoView({
        behavior: "smooth",
        block: "start",
    });
}

/* ────────────────────────────────────────────────────────────────
     ERROR TOAST
  ──────────────────────────────────────────────────────────────── */
function toggleMobileNav() {
    const toggle = document.getElementById("nav-toggle");
    const menu = document.getElementById("nav-menu");
    const isOpen = toggle.classList.toggle("active");
    menu.classList.toggle("open", isOpen);
    toggle.setAttribute("aria-expanded", String(isOpen));
}

function closeMobileNav() {
    const toggle = document.getElementById("nav-toggle");
    const menu = document.getElementById("nav-menu");
    toggle.classList.remove("active");
    menu.classList.remove("open");
    toggle.setAttribute("aria-expanded", "false");
}

document
    .getElementById("nav-toggle")
    .addEventListener("click", toggleMobileNav);
document.querySelectorAll(".nav-links a").forEach((link) => {
    link.addEventListener("click", () => {
        if (window.innerWidth <= 768) {
            closeMobileNav();
        }
    });
});

window.addEventListener("resize", () => {
    if (window.innerWidth > 768) {
        closeMobileNav();
    }
});

function showError(msg) {
    const toast = document.getElementById("error-toast");
    toast.textContent = "⚠ " + msg;
    toast.classList.add("show");
    setTimeout(() => toast.classList.remove("show"), 4000);
}

async function downloadReport() {
    console.log("[DEBUG] downloadReport() called");
    const payload = window.latestReportPayload;
    if (!payload || !payload.disease) {
        console.log("[DEBUG] No payload available");
        showError("Please generate a prediction before downloading a report.");
        return;
    }

    console.log("[DEBUG] Payload structure:", {
        crop: payload.crop,
        disease: payload.disease,
        confidence: payload.confidence,
        suggestion: payload.suggestion ?
            payload.suggestion.substring(0, 50) + "..." :
            "none",
        image_data_length: payload.image_data ? payload.image_data.length : 0,
    });

    const button = document.getElementById("download-report-btn");
    button.disabled = true;
    button.textContent = "Generating report…";

    try {
        // FIX for iPhone: Exclude large Base64 image_data on iOS to prevent
        // "The string did not match the expected pattern" error.
        // On iOS Safari, the 5-15MB Base64 string from camera captures causes
        // issues with JSON parsing and blob download handling.
        // The Flask route already handles empty image_data by showing a placeholder.
        const isIOSDevice = /iPad|iPhone|iPod/.test(navigator.userAgent);
        console.log("[DEBUG] Detected iOS device:", isIOSDevice);

        const payloadForServer = {
            crop: payload.crop,
            disease: payload.disease,
            confidence: payload.confidence,
            suggestion: payload.suggestion,
            image_data: isIOSDevice ? "" : payload.image_data || "",
        };

        if (isIOSDevice) {
            console.log(
                "[DEBUG] iOS detected - excluding image_data to prevent blob download issue",
            );
        }

        console.log("[DEBUG] Sending fetch request to /download_report");
        const jsonBody = JSON.stringify(payloadForServer);
        console.log("[DEBUG] JSON body size:", jsonBody.length, "bytes");

        const response = await fetch("/download_report", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: jsonBody,
        });

        console.log(
            "[DEBUG] Fetch completed. Status:",
            response.status,
            "OK:",
            response.ok,
        );
        console.log(
            "[DEBUG] Response headers - Content-Type:",
            response.headers.get("Content-Type"),
        );
        console.log(
            "[DEBUG] Response headers - Content-Disposition:",
            response.headers.get("Content-Disposition"),
        );
        console.log(
            "[DEBUG] Response headers - Content-Length:",
            response.headers.get("Content-Length"),
        );

        if (!response.ok) {
            console.log("[DEBUG] Response not OK, trying to parse error");
            const errorData = await response.json();
            console.log("[DEBUG] Error from server:", errorData);
            throw new Error(errorData.error || "Report generation failed.");
        }

        console.log("[DEBUG] Reading blob from response");
        const blob = await response.blob();
        console.log(
            "[DEBUG] Blob created. Type:",
            blob.type,
            "Size:",
            blob.size,
            "bytes",
        );

        console.log("[DEBUG] Creating object URL from blob");
        const url = URL.createObjectURL(blob);
        console.log("[DEBUG] Object URL created:", url);

        console.log("[DEBUG] Creating anchor element");
        const link = document.createElement("a");
        link.href = url;
        link.download = "AgriScan_Report.pdf";
        console.log("[DEBUG] Link download attribute set to:", link.download);

        console.log("[DEBUG] Appending link to document.body");
        document.body.appendChild(link);

        console.log("[DEBUG] Clicking link to trigger download");
        link.click();

        console.log("[DEBUG] Removing link from document");
        link.remove();

        console.log("[DEBUG] Revoking object URL");
        URL.revokeObjectURL(url);

        console.log("[DEBUG] Download completed successfully");
    } catch (err) {
        console.error(
            "[DEBUG] Error in downloadReport:",
            err.name,
            err.message,
            err.stack,
        );
        showError(err.message);
    } finally {
        button.disabled = false;
        button.textContent = "Download Report";
    }
}