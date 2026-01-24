const API_BASE = "";

async function loadHistory() {
    const res = await fetch("/history");
    const data = await res.json();

    const tbody = document.querySelector("#historyTable tbody");
    tbody.innerHTML = "";

    data.history.forEach(row => {
        const tr = document.createElement("tr");

        tr.innerHTML = `
            <td>${row.result_type}</td>
            <td>${row.comment || row.url || "-"}</td>
            <td>${row.total_comments || "-"}</td>
            <td>${row.sentiment || "-"}</td>
            <td>${row.type_ || "-"}</td>
            <td>${row.date}</td>
            <td>
                ${row.sentiment_distribution_plot ? `
                <button onclick="openPlots(
                    '${row.sentiment_distribution_plot}',
                    '${row.sentiment_confidence_plot}',
                    '${row.type_distribution_plot}',
                    '${row.type_confidence_plot}'
                )">View</button>` : "-"}
            </td>
        `;

        tbody.appendChild(tr);
    });
}

function openPlots(sd, sp, td, tp) {
    const params = new URLSearchParams({ sd, sp, td, tp });
    window.location.href = `/plots-page?${params.toString()}`
}

loadHistory();
