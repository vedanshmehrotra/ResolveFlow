document.addEventListener('DOMContentLoaded', () => {
    // Mobile Sidebar Toggle
    const hamburger = document.getElementById('mobile-hamburger');
    const sidebar = document.getElementById('sidebar');
    const backdrop = document.getElementById('sidebar-backdrop');

    if (hamburger && sidebar && backdrop) {
        hamburger.addEventListener('click', () => {
            sidebar.classList.toggle('-translate-x-full');
            backdrop.classList.toggle('hidden');
        });

        backdrop.addEventListener('click', () => {
            sidebar.classList.add('-translate-x-full');
            backdrop.classList.add('hidden');
        });
    }

    // Client-side Table Search (Audit Log)
    const searchInput = document.getElementById('audit-search');
    if (searchInput) {
        searchInput.addEventListener('input', (e) => {
            const term = e.target.value.toLowerCase();
            const rows = document.querySelectorAll('#audit-table tbody tr');
            rows.forEach(row => {
                const text = row.textContent.toLowerCase();
                row.style.display = text.includes(term) ? '' : 'none';
            });
        });
    }

    // Client-side Tabs (Operations Inbox)
    const tabs = document.querySelectorAll('.filter-tab');
    if (tabs.length > 0) {
        tabs.forEach(tab => {
            tab.addEventListener('click', () => {
                tabs.forEach(t => t.classList.remove('active'));
                tab.classList.add('active');
                
                const filter = tab.dataset.filter.toUpperCase();
                const rows = document.querySelectorAll('.assignment-row');
                rows.forEach(row => {
                    if (filter === 'ALL') {
                        row.style.display = '';
                    } else {
                        const status = row.dataset.status.toUpperCase();
                        row.style.display = (status === filter) ? '' : 'none';
                    }
                });
            });
        });
    }
});

// Inline Validation Helper
function showInputError(inputEl, message) {
    inputEl.classList.add('border-danger');
    let errorP = inputEl.nextElementSibling;
    if (!errorP || !errorP.classList.contains('error-text')) {
        errorP = document.createElement('p');
        errorP.className = 'error-text text-danger text-xs mt-1';
        inputEl.parentNode.insertBefore(errorP, inputEl.nextSibling);
    }
    errorP.textContent = message;
    
    inputEl.addEventListener('focus', () => {
        inputEl.classList.remove('border-danger');
        if (errorP) errorP.remove();
    }, { once: true });
}
