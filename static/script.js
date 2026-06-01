function showToast(msg, type = 'success') {
  const toast = document.getElementById('toast');
  toast.textContent = msg;
  toast.className = `toast ${type} show`;
  setTimeout(() => { toast.className = 'toast'; }, 3500);
}

function setLoading(btnId, loading) {
  const btn = document.getElementById(btnId);
  btn.querySelector('.btn-text').style.display = loading ? 'none' : '';
  btn.querySelector('.btn-loader').style.display = loading ? '' : 'none';
  btn.disabled = loading;
}

function registerUser() {
  const name = document.getElementById('name').value.trim();
  const id = document.getElementById('id').value.trim();

  if (!name || !id) {
    showToast('Please fill in both Name and User ID.', 'error');
    return;
  }

  setLoading('register-btn', true);

  fetch('/register', {
    method: 'POST',
    headers: { 'Content-Type': 'application/json' },
    body: JSON.stringify({ name, id })
  })
    .then(r => r.json())
    .then(data => {
      showToast(data.message || 'Registration complete.', 'success');
      document.getElementById('name').value = '';
      document.getElementById('id').value = '';
    })
    .catch(() => showToast('Registration failed. Check server.', 'error'))
    .finally(() => setLoading('register-btn', false));
}

function recognizeFace() {
  setLoading('recognize-btn', true);

  const display = document.getElementById('result-display');
  display.innerHTML = '<span class="result-idle">Scanning...</span>';

  fetch('/recognize')
    .then(r => r.json())
    .then(data => {
      const person = data.person || 'Unknown';
      if (person.toLowerCase() === 'unknown') {
        display.innerHTML = `<span class="result-unknown">⚠ UNKNOWN PERSON</span>`;
        showToast('No match found.', 'error');
      } else {
        display.innerHTML = `<span class="result-name">✔ ${person}</span>`;
        showToast(`Identified: ${person}`, 'success');
      }
    })
    .catch(() => {
      display.innerHTML = '<span class="result-unknown">⚠ ERROR</span>';
      showToast('Recognition failed. Check server.', 'error');
    })
    .finally(() => setLoading('recognize-btn', false));
}
