const backendBase = "https://trackassessment-production-4b15.up.railway.app"; // Your backend URL
const DASHBOARD_PAGE = 'user-dashboard.html';
const ADMIN_PAGE = 'questions-all.html?category=' + encodeURIComponent('All Questions');

function showMessage(msg, isError = false){
  const el = document.getElementById('authMessage');
  if (!el) return;
  el.textContent = msg || '';
  el.style.color = isError ? '#dc2626' : '';
}

async function authWithPassword(email, password){
  try{
    const url = `${backendBase}/auth`;
    const res = await fetch(url, {
      method: 'POST',
      headers: { 'Content-Type': 'application/json', 'Accept': 'application/json' },
      body: JSON.stringify({ email, password }),
    });
    if (!res.ok){
      const t = await res.text();
      throw new Error(`${res.status} ${res.statusText} - ${t}`);
    }
    return await res.json();
  }catch(err){
    console.warn('Backend auth failed:', err);
    return { error: err.message };
  }
}

async function fetchProfileFromBackend(email){
  try{
    const purl = `${backendBase}/user-profile?email=${encodeURIComponent(email)}`;
    const pres = await fetch(purl, { headers: { Accept: 'application/json' } });
    if (!pres.ok) return null;
    const pj = await pres.json();
    if (!pj || pj.error || !Object.keys(pj).length) return null;
    return pj;
  }catch(error){
    console.warn('Failed to fetch backend profile:', error);
    return null;
  }
}

function saveSession(profile){
  const safe = normalizeProfile(profile, profile?.email || '');
  sessionStorage.setItem('user_email', safe.email || '');
  sessionStorage.setItem('user_id', safe.user_id !== undefined && safe.user_id !== null ? String(safe.user_id) : '');
  sessionStorage.setItem('user_name', safe.full_name || safe.name || '');
  sessionStorage.setItem('user_profile', JSON.stringify(safe));
}

function normalizeProfile(row, emailFallback = ''){
  const safe = row && typeof row === 'object' ? row : {};
  const fullName = safe.full_name || safe.name || safe.fullName || safe.student_name || safe.username || '';
  const email = safe.email || emailFallback || '';
  return {
    ...safe,
    user_id: safe.user_id ?? safe.id ?? safe.userId ?? safe.uuid ?? '',
    email,
    full_name: fullName,
    name: safe.name || fullName,
    usertype: (safe.usertype || safe.user_type || safe.type || 'user').toLowerCase()
  };
}

document.getElementById('loginForm').addEventListener('submit', async (ev) => {
  ev.preventDefault();
  const email = document.getElementById('loginEmail').value.trim();
  const password = document.getElementById('loginPassword').value || '';
  if (!email || !password) return showMessage('Please enter email and password.', true);

  showMessage('Authenticating...');

  let authPayload = await authWithPassword(email, password);
  let directProfile = null;

  if (!authPayload || authPayload.error){
    try{
      directProfile = await authDirectFromSupabase(email, password);
    }catch(error){
      console.warn('Supabase direct auth failed:', error);
    }

    if (!directProfile){
      showMessage('Login failed. Check your email or password.', true);
      return;
    }
  }

  const profile = directProfile
    ? normalizeProfile(directProfile, email)
    : await resolveLoginProfile(email, password, authPayload);

  if (!profile.email) profile.email = email;
  saveSession(profile);
  showMessage('Login successful. Redirecting...');
  redirectByType(profile);
});

const showPw = document.getElementById('showPassword');
if (showPw){
  showPw.addEventListener('change', () => {
    const p = document.getElementById('loginPassword');
    if (p) p.type = showPw.checked ? 'text' : 'password';
  });
}

function redirectByType(profile){
  const usertype = (profile.usertype || profile.user_type || profile.type || 'user').toLowerCase();
  window.location.href = usertype === 'admin' ? ADMIN_PAGE : DASHBOARD_PAGE;
}