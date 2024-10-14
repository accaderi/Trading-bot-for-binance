function toggleSidePanel() {
    const sidePanel = document.getElementById('sidePanel');
    const mainContent = document.getElementById('mainContent');
    sidePanel.classList.toggle('hidden');
    mainContent.classList.toggle('expanded');
}
