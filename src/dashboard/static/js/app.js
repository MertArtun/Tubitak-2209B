/**
 * Student Attention Detection Dashboard
 * Frontend logic: webcam capture, API communication, chart rendering.
 */

(function () {
    "use strict";

    // ── State ───────────────────────────────────────────────────────────
    const state = {
        cameraStream: null,
        analyzeInterval: null,
        activeSessionId: null,
        frameCount: 0,
        fpsTimestamps: [],
        timelineData: [],
        emotionCounts: { negative: 0, neutral: 0, positive: 0 },
        studentScores: {},       // { name: avgScore }
        capturedBlobs: [],       // registration photos
        regStream: null,
    };

    // ── DOM refs ────────────────────────────────────────────────────────
    const $ = (sel) => document.querySelector(sel);
    const video            = $("#webcam");
    const overlayCanvas    = $("#overlay-canvas");
    const overlayCtx       = overlayCanvas.getContext("2d");
    const btnStartCam      = $("#btn-start-camera");
    const btnStopCam       = $("#btn-stop-camera");
    const btnStartSession  = $("#btn-start-session");
    const btnStopSession   = $("#btn-stop-session");
    const btnExport        = $("#btn-export");
    const btnOpenRegister  = $("#btn-open-register");
    const btnCloseModal    = $("#btn-close-modal");
    const btnCapture       = $("#btn-capture");
    const btnClearCaptures = $("#btn-clear-captures");
    const btnRegister      = $("#btn-register");
    const registerModal    = $("#register-modal");
    const regWebcam        = $("#reg-webcam");
    const regCanvas        = $("#reg-canvas");
    const capturedImgs     = $("#captured-images");
    const regStatus        = $("#reg-status");
    const fpsIndicator     = $("#fps-indicator");
    const sessionBadge     = $("#session-badge");
    const gaugeValue       = $("#gauge-value");
    const gaugeFill        = $("#gauge-fill");
    const studentMetrics   = $("#student-metrics");
    const studentList      = $("#student-list");
    const themeToggle      = $("#theme-toggle");
    const themeIcon        = $("#theme-icon");

    // ── Charts ──────────────────────────────────────────────────────────
    let chartTimeline, chartEmotions, chartStudents;

    function initCharts() {
        const fontColor = getComputedStyle(document.documentElement)
            .getPropertyValue("--text-primary").trim() || "#333";

        Chart.defaults.color = fontColor;

        chartTimeline = new Chart($("#chart-timeline"), {
            type: "line",
            data: {
                labels: [],
                datasets: [{
                    label: "Sinif Ortalamasi",
                    data: [],
                    borderColor: "#1976D2",
                    backgroundColor: "rgba(25,118,210,0.1)",
                    fill: true,
                    tension: 0.3,
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { min: 0, max: 1 } },
                plugins: { legend: { display: false } },
                animation: { duration: 200 },
            },
        });

        chartEmotions = new Chart($("#chart-emotions"), {
            type: "doughnut",
            data: {
                labels: ["Negatif", "Notr", "Pozitif"],
                datasets: [{
                    data: [0, 0, 0],
                    backgroundColor: ["#F44336", "#FF9800", "#4CAF50"],
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: { duration: 200 },
            },
        });

        chartStudents = new Chart($("#chart-students"), {
            type: "bar",
            data: {
                labels: [],
                datasets: [{
                    label: "Dikkat Skoru",
                    data: [],
                    backgroundColor: "#1976D2",
                }],
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                scales: { y: { min: 0, max: 1 } },
                plugins: { legend: { display: false } },
                animation: { duration: 200 },
            },
        });
    }

    // ── Theme ───────────────────────────────────────────────────────────
    function applyTheme(dark) {
        document.documentElement.setAttribute("data-theme", dark ? "dark" : "light");
        themeIcon.textContent = dark ? "\u2600" : "\u263E";
        localStorage.setItem("theme", dark ? "dark" : "light");
    }

    themeToggle.addEventListener("click", () => {
        const isDark = document.documentElement.getAttribute("data-theme") === "dark";
        applyTheme(!isDark);
    });

    // Restore saved theme
    if (localStorage.getItem("theme") === "dark") {
        applyTheme(true);
    }

    // ── Toast ───────────────────────────────────────────────────────────
    function showToast(msg, type) {
        type = type || "info";
        const el = document.createElement("div");
        el.className = "toast toast-" + type;
        el.textContent = msg;
        $("#toast-container").appendChild(el);
        setTimeout(() => el.remove(), 3500);
    }

    // ── Camera ──────────────────────────────────────────────────────────
    async function startCamera() {
        try {
            state.cameraStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 640, height: 480 },
                audio: false,
            });
            video.srcObject = state.cameraStream;
            btnStartCam.disabled = true;
            btnStopCam.disabled = false;
            showToast("Kamera baslatildi.", "success");
        } catch (err) {
            showToast("Kamera erisimi reddedildi: " + err.message, "error");
        }
    }

    function stopCamera() {
        if (state.cameraStream) {
            state.cameraStream.getTracks().forEach((t) => t.stop());
            state.cameraStream = null;
        }
        video.srcObject = null;
        stopAnalyzeLoop();
        btnStartCam.disabled = false;
        btnStopCam.disabled = true;
    }

    btnStartCam.addEventListener("click", startCamera);
    btnStopCam.addEventListener("click", stopCamera);

    // ── Frame capture & analysis ────────────────────────────────────────
    function captureFrame() {
        if (!video.videoWidth) return null;
        const canvas = document.createElement("canvas");
        canvas.width = video.videoWidth;
        canvas.height = video.videoHeight;
        canvas.getContext("2d").drawImage(video, 0, 0);
        return canvas.toDataURL("image/jpeg", 0.8);
    }

    async function analyzeFrame() {
        const dataUrl = captureFrame();
        if (!dataUrl) return;

        try {
            const resp = await fetch("/api/analyze", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ image: dataUrl }),
            });
            if (!resp.ok) throw new Error("HTTP " + resp.status);
            const data = await resp.json();
            handleAnalyzeResult(data);
        } catch (err) {
            console.error("Analyze error:", err);
        }
    }

    function startAnalyzeLoop() {
        if (state.analyzeInterval) return;
        state.analyzeInterval = setInterval(analyzeFrame, 500); // 2 FPS
    }

    function stopAnalyzeLoop() {
        if (state.analyzeInterval) {
            clearInterval(state.analyzeInterval);
            state.analyzeInterval = null;
        }
    }

    // ── Handle analysis result ──────────────────────────────────────────
    function handleAnalyzeResult(data) {
        // FPS tracking
        const now = Date.now();
        state.fpsTimestamps.push(now);
        state.fpsTimestamps = state.fpsTimestamps.filter((t) => now - t < 2000);
        const fps = Math.round(state.fpsTimestamps.length / 2);
        fpsIndicator.textContent = fps + " FPS";

        const results = data.results || [];

        // Draw bounding boxes on overlay
        drawOverlay(results);

        if (!results.length) return;

        // Compute class average
        let totalScore = 0;
        results.forEach((r) => {
            totalScore += r.attention_score;

            // Emotion counts
            if (r.emotion in state.emotionCounts) {
                state.emotionCounts[r.emotion]++;
            }

            // Per-student tracking
            const name = r.name || "Bilinmeyen";
            if (!state.studentScores[name]) {
                state.studentScores[name] = { total: 0, count: 0 };
            }
            state.studentScores[name].total += r.attention_score;
            state.studentScores[name].count += 1;
        });

        const classAvg = totalScore / results.length;

        // Update gauge
        updateGauge(classAvg);

        // Update student metric cards
        updateStudentMetrics(results);

        // Timeline
        const label = new Date(data.timestamp).toLocaleTimeString("tr-TR");
        state.timelineData.push({ label: label, value: classAvg });
        if (state.timelineData.length > 60) state.timelineData.shift();

        updateCharts();
    }

    // ── Overlay drawing ─────────────────────────────────────────────────
    function drawOverlay(results) {
        overlayCanvas.width = video.videoWidth || 640;
        overlayCanvas.height = video.videoHeight || 480;
        overlayCtx.clearRect(0, 0, overlayCanvas.width, overlayCanvas.height);

        results.forEach((r) => {
            const [x1, y1, x2, y2] = r.bbox;
            const color = r.attention_level === "focused" ? "#4CAF50"
                        : r.attention_level === "moderate" ? "#FF9800"
                        : "#F44336";

            overlayCtx.strokeStyle = color;
            overlayCtx.lineWidth = 2;
            overlayCtx.strokeRect(x1, y1, x2 - x1, y2 - y1);

            // Label
            const labelText = (r.name || "?") + " | " + r.emotion + " " +
                              Math.round(r.attention_score * 100) + "%";
            overlayCtx.font = "12px sans-serif";
            const tw = overlayCtx.measureText(labelText).width;
            overlayCtx.fillStyle = color;
            overlayCtx.fillRect(x1, y1 - 18, tw + 8, 18);
            overlayCtx.fillStyle = "#fff";
            overlayCtx.fillText(labelText, x1 + 4, y1 - 5);
        });
    }

    // ── Gauge update ────────────────────────────────────────────────────
    function updateGauge(score) {
        const pct = Math.round(score * 100);
        gaugeValue.textContent = pct + "%";

        // Arc length: ~251.2 for the semi-circle path
        const maxLen = 251.2;
        const fill = maxLen * score;
        gaugeFill.setAttribute("stroke-dasharray", fill + " " + maxLen);

        // Color
        const color = score > 0.6 ? "#4CAF50" : score > 0.35 ? "#FF9800" : "#F44336";
        gaugeFill.setAttribute("stroke", color);
    }

    // ── Student metric cards ────────────────────────────────────────────
    function updateStudentMetrics(results) {
        let html = "";
        results.forEach((r) => {
            const name = r.name || "Bilinmeyen";
            const initials = name.split(" ").map((w) => w[0]).join("").substring(0, 2);
            const levelColor = r.attention_level === "focused" ? "var(--focused)"
                             : r.attention_level === "moderate" ? "var(--moderate)"
                             : "var(--distracted)";
            const emotionTr = { positive: "Pozitif", negative: "Negatif", neutral: "Notr" };

            html += '<div class="student-metric-card">' +
                '<div class="avatar">' + initials.toUpperCase() + '</div>' +
                '<div class="info">' +
                    '<div class="name">' + escapeHtml(name) + '</div>' +
                    '<div class="detail">' + (emotionTr[r.emotion] || r.emotion) +
                    ' &middot; ' + Math.round(r.attention_score * 100) + '%</div>' +
                '</div>' +
                '<div class="level-indicator" style="background:' + levelColor + '"></div>' +
            '</div>';
        });
        studentMetrics.innerHTML = html || '<p class="text-muted">Analiz bekleniyor...</p>';
    }

    // ── Chart updates ───────────────────────────────────────────────────
    function updateCharts() {
        // Timeline
        chartTimeline.data.labels = state.timelineData.map((d) => d.label);
        chartTimeline.data.datasets[0].data = state.timelineData.map((d) => d.value);
        chartTimeline.update("none");

        // Emotions
        chartEmotions.data.datasets[0].data = [
            state.emotionCounts.negative,
            state.emotionCounts.neutral,
            state.emotionCounts.positive,
        ];
        chartEmotions.update("none");

        // Students bar chart
        const names = Object.keys(state.studentScores);
        const avgs = names.map((n) => {
            const s = state.studentScores[n];
            return s.count ? s.total / s.count : 0;
        });
        chartStudents.data.labels = names;
        chartStudents.data.datasets[0].data = avgs;
        chartStudents.update("none");
    }

    // ── Session controls ────────────────────────────────────────────────
    btnStartSession.addEventListener("click", async () => {
        const name = $("#session-name").value.trim();
        if (!name) { showToast("Oturum adi girin.", "error"); return; }
        const mode = $("#session-mode").value;

        try {
            const resp = await fetch("/api/sessions/start", {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ name: name, mode: mode }),
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || "Hata");

            state.activeSessionId = data.session_id;
            sessionBadge.textContent = "Oturum #" + data.session_id;
            sessionBadge.className = "badge badge-focused";
            btnStartSession.disabled = true;
            btnStopSession.disabled = false;

            // Start analyzing if camera is on
            if (state.cameraStream) startAnalyzeLoop();

            // Reset chart data for new session
            state.timelineData = [];
            state.emotionCounts = { negative: 0, neutral: 0, positive: 0 };
            state.studentScores = {};

            showToast("Oturum baslatildi.", "success");
        } catch (err) {
            showToast("Oturum baslatilamadi: " + err.message, "error");
        }
    });

    btnStopSession.addEventListener("click", async () => {
        if (!state.activeSessionId) return;
        try {
            const resp = await fetch("/api/sessions/" + state.activeSessionId + "/stop", {
                method: "POST",
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || "Hata");

            stopAnalyzeLoop();
            state.activeSessionId = null;
            sessionBadge.textContent = "Oturum Yok";
            sessionBadge.className = "badge badge-secondary";
            btnStartSession.disabled = false;
            btnStopSession.disabled = true;

            showToast("Oturum sonlandirildi.", "success");
        } catch (err) {
            showToast("Oturum durdurulamadi: " + err.message, "error");
        }
    });

    // ── Student list ────────────────────────────────────────────────────
    async function loadStudents() {
        try {
            const resp = await fetch("/api/students");
            const data = await resp.json();
            const students = data.students || [];

            if (!students.length) {
                studentList.innerHTML = '<p class="text-muted">Henuz ogrenci kaydedilmedi.</p>';
                return;
            }

            let html = "";
            students.forEach((s) => {
                const initials = s.name.split(" ").map((w) => w[0]).join("").substring(0, 2);
                html += '<div class="student-list-item" data-id="' + s.id + '">' +
                    '<div class="avatar-sm">' + initials.toUpperCase() + '</div>' +
                    '<div class="student-info">' + escapeHtml(s.name) + '</div>' +
                '</div>';
            });
            studentList.innerHTML = html;
        } catch (err) {
            console.error("Failed to load students:", err);
        }
    }

    // ── Export ───────────────────────────────────────────────────────────
    btnExport.addEventListener("click", () => {
        let url = "/api/export/excel";
        if (state.activeSessionId) {
            url += "?session_id=" + state.activeSessionId;
        }
        window.location.href = url;
    });

    // ── Registration modal ──────────────────────────────────────────────
    btnOpenRegister.addEventListener("click", async () => {
        registerModal.classList.remove("hidden");
        try {
            state.regStream = await navigator.mediaDevices.getUserMedia({
                video: { width: 320, height: 240 },
                audio: false,
            });
            regWebcam.srcObject = state.regStream;
        } catch (err) {
            showToast("Kamera erisimi reddedildi.", "error");
        }
    });

    btnCloseModal.addEventListener("click", closeRegModal);
    registerModal.querySelector(".modal-backdrop").addEventListener("click", closeRegModal);

    function closeRegModal() {
        registerModal.classList.add("hidden");
        if (state.regStream) {
            state.regStream.getTracks().forEach((t) => t.stop());
            state.regStream = null;
        }
        regWebcam.srcObject = null;
        state.capturedBlobs = [];
        capturedImgs.innerHTML = "";
        regStatus.textContent = "";
        $("#reg-name").value = "";
        $("#reg-email").value = "";
    }

    btnCapture.addEventListener("click", () => {
        if (!regWebcam.videoWidth) return;
        regCanvas.width = regWebcam.videoWidth;
        regCanvas.height = regWebcam.videoHeight;
        regCanvas.getContext("2d").drawImage(regWebcam, 0, 0);

        regCanvas.toBlob((blob) => {
            state.capturedBlobs.push(blob);
            const img = document.createElement("img");
            img.src = URL.createObjectURL(blob);
            capturedImgs.appendChild(img);
        }, "image/jpeg", 0.9);
    });

    btnClearCaptures.addEventListener("click", () => {
        state.capturedBlobs = [];
        capturedImgs.innerHTML = "";
    });

    btnRegister.addEventListener("click", async () => {
        const name = $("#reg-name").value.trim();
        if (!name) { showToast("Ad Soyad girin.", "error"); return; }
        if (!state.capturedBlobs.length) { showToast("En az bir fotograf cekin.", "error"); return; }

        regStatus.textContent = "Kaydediliyor...";
        const formData = new FormData();
        formData.append("name", name);
        const email = $("#reg-email").value.trim();
        if (email) formData.append("email", email);
        state.capturedBlobs.forEach((blob, i) => {
            formData.append("images", blob, "photo_" + i + ".jpg");
        });

        try {
            const resp = await fetch("/api/students/register", {
                method: "POST",
                body: formData,
            });
            const data = await resp.json();
            if (!resp.ok) throw new Error(data.error || "Kayit hatasi");

            showToast(data.message, "success");
            closeRegModal();
            loadStudents();
        } catch (err) {
            regStatus.textContent = "";
            showToast("Kayit basarisiz: " + err.message, "error");
        }
    });

    // ── Utilities ───────────────────────────────────────────────────────
    function escapeHtml(str) {
        const div = document.createElement("div");
        div.textContent = str;
        return div.innerHTML;
    }

    // ── Init ────────────────────────────────────────────────────────────
    initCharts();
    loadStudents();
})();
