/* Login page — form submission, error handling */
(function() {
    var form = document.getElementById('login-form');
    var btn = document.getElementById('btn-login');
    var errorBox = document.getElementById('login-error');
    var errorText = document.getElementById('login-error-text');
    var usernameEl = document.getElementById('username');
    var passwordEl = document.getElementById('password');
    var rememberEl = document.getElementById('remember-me');

    function showError(msg) {
        errorText.textContent = msg;
        errorBox.classList.remove('hidden');
    }

    function hideError() {
        errorBox.classList.add('hidden');
    }

    function setLoading(loading) {
        btn.disabled = loading;
        btn.textContent = loading ? 'Signing in...' : 'Sign In';
    }

    form.addEventListener('submit', function(e) {
        e.preventDefault();
        hideError();

        var username = usernameEl.value.trim();
        var password = passwordEl.value;

        if (!username || !password) {
            showError('Please enter username and password');
            return;
        }

        setLoading(true);

        fetch('/api/v1/auth/login', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({
                username: username,
                password: password,
                remember_me: rememberEl.checked
            })
        })
        .then(function(res) {
            if (!res.ok) {
                return res.json().then(function(data) {
                    throw new Error(data.detail || 'Invalid credentials');
                }, function() {
                    throw new Error('Invalid credentials');
                });
            }
            return res.json();
        })
        .then(function(data) {
            // Success — redirect to home
            window.location.href = '/';
        })
        .catch(function(err) {
            showError(err.message || 'Login failed');
            setLoading(false);
        });
    });

    // Focus first input
    if (usernameEl && !usernameEl.value) {
        usernameEl.focus();
    }
})();
