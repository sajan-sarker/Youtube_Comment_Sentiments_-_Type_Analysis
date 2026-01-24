const params = new URLSearchParams(window.location.search);

["sd", "sp", "td", "tp"].forEach(id => {
    let path = params.get(id);

    if (!path) return;

    if (path.startsWith("app/plots/")) {
        path = path.replace("app/plots/", "/plots/");
    }

    document.getElementById(id).src = path;
});
