const API_BASE = "";

/* -------- Single Comment -------- */
async function analyzeComment() {
    const comment = document.getElementById("commentInput").value;

    const res = await fetch(`${API_BASE}/predict`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ comment })
    });

    const data = await res.json();

    document.getElementById("commentResult").innerHTML = `
        <p><b>Sentiment:</b> ${data.sentiment} (${data.sentiment_confidence}%)</p>
        <p><b>Type:</b> ${data.comment_type} (${data.type_confidence}%)</p>
    `;
}

/* -------- URL Analysis -------- */
async function analyzeURL() {
    const url = document.getElementById("urlInput").value;

    // Reset old results
    document.getElementById("totalCommentsBox").style.display = "none";
    document.getElementById("homePlots").style.display = "none";

    showLoader();

    try {
        const res = await fetch(`${API_BASE}/analyze`, {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ url })
        });

        const data = await res.json();

        hideLoader();

        /* Show total comments */
        const totalBox = document.getElementById("totalCommentsBox");
        totalBox.style.display = "flex";
        totalBox.innerHTML = `Total Comments Analyzed: ${data.total_comments}`;

        /* Show plots */
        document.getElementById("homePlots").style.display = "grid";

        document.getElementById("home_sd").src = `${API_BASE}${data.sentiment_distribution_plot}`;
        document.getElementById("home_sp").src = `${API_BASE}${data.sentiment_confidence_plot}`;
        document.getElementById("home_td").src = `${API_BASE}${data.type_distribution_plot}`;
        document.getElementById("home_tp").src = `${API_BASE}${data.type_confidence_plot}`;

    } catch (error) {
        hideLoader();
        alert("Error analyzing video. Please try again.");
        console.error(error);
    }
}

/* -------- Loader helpers -------- */
function showLoader() {
    document.getElementById("loadingContainer").style.display = "block";
}

function hideLoader() {
    document.getElementById("loadingContainer").style.display = "none";
}
