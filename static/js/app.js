document.addEventListener('DOMContentLoaded', function() {
    // System control
    document.getElementById('toggle-system').addEventListener('click', function() {
        const action = this.textContent.trim().toLowerCase();
        fetch(`/api/system/${action}`, { method: 'POST' })
            .then(response => response.json())
            .then(data => {
                if (data.status === 'success') {
                    location.reload();
                }
            });
    });

    // Camera controls
    document.querySelectorAll('.btn-start').forEach(btn => {
        btn.addEventListener('click', function() {
            const cameraName = this.dataset.camera;
            fetch(`/api/camera/${cameraName}/start`, { method: 'POST' })
                .then(response => response.json())
                .then(data => {
                    if (data.status === 'success') {
                        location.reload();
                    }
                });
        });
    });
});