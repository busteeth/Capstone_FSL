document.addEventListener('DOMContentLoaded', () => {
  const statusEl   = document.getElementById('status');
  const btn        = document.getElementById('toggleRecognition');
  const signEl     = document.getElementById('detected-sign');
  const confEl     = document.getElementById('confidence');

  let isActive = false;

  const updateUI = () => {
    btn.innerHTML = isActive
      ? '<i class="fas fa-pause"></i> Pause'
      : '<i class="fas fa-play"></i> Start';
    statusEl.textContent = isActive ? 'Active' : 'Camera ready';
    statusEl.classList.toggle('live', isActive);
  };

  btn.addEventListener('click', () => {
    isActive = !isActive;
    updateUI();
  });

  // Poll for latest detected letter
  setInterval(async () => {
    if (!isActive) return;

    try {
      const res = await fetch('/get_prediction');
      const data = await res.json();

      const currentSign = data.sign || "—";

      signEl.textContent = currentSign;

      if (currentSign !== "—") {
        confEl.textContent = `Confidence: ${data.confidence}%`;
        confEl.style.color = "#aaffaa";
      } else {
        confEl.textContent = "Confidence: —";
        confEl.style.color = "#ccc";
      }
    } catch (err) {
      console.error("Failed to fetch prediction:", err);
    }
  }, 600);

  // Auto-start after short delay (optional – remove if you don't want it)
  setTimeout(() => {
    if (!isActive) btn.click();
  }, 1200);
});