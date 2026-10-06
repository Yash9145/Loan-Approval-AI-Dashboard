const form = document.getElementById("predictionForm");
const result = document.getElementById("result");
const label = document.getElementById("resultLabel");
const message = document.getElementById("resultMessage");
const approvalPct = document.getElementById("approvalPct");
const rejectionPct = document.getElementById("rejectionPct");
const approvalBar = document.getElementById("approvalBar");
const rejectionBar = document.getElementById("rejectionBar");
const risk = document.getElementById("risk");
const icon = document.getElementById("resultIcon");

form.addEventListener("submit", async (event) => {
  event.preventDefault();

  const data = Object.fromEntries(new FormData(form).entries());

  try {
    const response = await fetch("/api/predict", {
      method: "POST",
      headers: {"Content-Type": "application/json"},
      body: JSON.stringify(data)
    });

    const output = await response.json();
    if (!response.ok) throw new Error(output.error || "Prediction failed");

    result.classList.remove("hidden");
    label.textContent = output.label;
    message.textContent = output.message;
    approvalPct.textContent = `${output.approval_probability}%`;
    rejectionPct.textContent = `${output.rejection_probability}%`;
    approvalBar.style.width = `${output.approval_probability}%`;
    rejectionBar.style.width = `${output.rejection_probability}%`;
    risk.textContent = `${output.risk} RISK`;
    icon.textContent = output.approved ? "✓" : "×";

    result.scrollIntoView({behavior: "smooth", block: "center"});
  } catch (error) {
    alert(error.message);
  }
});
