document.addEventListener("DOMContentLoaded", () => {

    // --- NEURAL NEXUS (VISUALS) ---
    class NeuralNetwork {
        constructor() {
            this.canvas = document.getElementById('neural-canvas');
            if (!this.canvas) return;
            this.ctx = this.canvas.getContext('2d');
            this.particles = [];
            this.baseParticleCount = 80;
            this.connectDist = 160;
            this.mouseDist = 250;
            this.mouse = { x: -1000, y: -1000 };
            this.resize();
            let resizeTimeout;
            window.addEventListener('resize', () => { clearTimeout(resizeTimeout); resizeTimeout = setTimeout(() => this.resize(), 100); });
            document.addEventListener('mousemove', (e) => { this.mouse.x = e.clientX; this.mouse.y = e.clientY; });
            this.animate();
        }
        resize() {
            this.canvas.width = window.innerWidth;
            this.canvas.height = window.innerHeight;
            const area = this.canvas.width * this.canvas.height;
            const density = window.innerWidth < 768 ? 15000 : 9000;
            this.particleCount = Math.floor(area / density);
            this.init();
        }
        init() {
            this.particles = [];
            // Palette for Light Mode (Colorful)
            const palette = ['hsla(0,100%,60%,1)', 'hsla(30,100%,60%,1)', 'hsla(60,100%,60%,1)', 'hsla(120,100%,60%,1)', 'hsla(220,100%,60%,1)', 'hsla(270,100%,60%,1)'];
            for (let i = 0; i < this.particleCount; i++) {
                this.particles.push({
                    x: Math.random() * this.canvas.width, y: Math.random() * this.canvas.height,
                    vx: (Math.random() - 0.5) * 0.8, vy: (Math.random() - 0.5) * 0.8,
                    size: Math.random() * 2.5 + 2,
                    color: palette[Math.floor(Math.random() * palette.length)]
                });
            }
        }
        animate() {
            this.ctx.clearRect(0, 0, this.canvas.width, this.canvas.height);
            const isLight = document.documentElement.classList.contains('light-mode');
            const lineBase = isLight ? '0, 0, 0' : '0, 243, 255';

            this.particles.forEach((p, i) => {
                p.x += p.vx; p.y += p.vy;
                if (p.x < 0 || p.x > this.canvas.width) p.vx *= -1;
                if (p.y < 0 || p.y > this.canvas.height) p.vy *= -1;

                // --- CURSOR INTERACTION (NO WARP) ---
                const dx = this.mouse.x - p.x;
                const dy = this.mouse.y - p.y;
                const dist = Math.sqrt(dx * dx + dy * dy);

                // Just draw the line, NO physics movement
                if (dist < this.mouseDist) {
                    this.ctx.beginPath();
                    const opacity = 1 - (dist / this.mouseDist);
                    this.ctx.strokeStyle = `rgba(${lineBase}, ${opacity})`;
                    this.ctx.lineWidth = 1.5;
                    this.ctx.moveTo(this.mouse.x, this.mouse.y);
                    this.ctx.lineTo(p.x, p.y);
                    this.ctx.stroke();
                }
                // ------------------------------------

                this.ctx.beginPath(); this.ctx.arc(p.x, p.y, p.size, 0, Math.PI * 2);

                if (isLight) {
                    this.ctx.fillStyle = p.color; // Colorful in Light Mode
                    this.ctx.shadowBlur = 5;
                    this.ctx.shadowColor = p.color;
                } else {
                    this.ctx.fillStyle = '#ffffff'; // White in Dark Mode
                    this.ctx.shadowBlur = 10;
                    this.ctx.shadowColor = '#ffffff';
                }

                this.ctx.fill(); this.ctx.shadowBlur = 0;
                for (let j = i + 1; j < this.particles.length; j++) {
                    const p2 = this.particles[j];
                    const dx2 = p.x - p2.x; const dy2 = p.y - p2.y;
                    const dist2 = Math.sqrt(dx2 * dx2 + dy2 * dy2);
                    if (dist2 < this.connectDist) {
                        this.ctx.beginPath(); const opacity = 1 - (dist2 / this.connectDist);
                        this.ctx.strokeStyle = `rgba(${lineBase}, ${opacity * 0.4})`;
                        this.ctx.lineWidth = 1.5; this.ctx.moveTo(p.x, p.y); this.ctx.lineTo(p2.x, p2.y);
                        this.ctx.stroke();
                    }
                }
            });
            requestAnimationFrame(() => this.animate());
        }
    }
    new NeuralNetwork();

    // --- APP LOGIC ---
    const API_URL = "https://nexus-neural-search.onrender.com";
    let AUTH_TOKEN = localStorage.getItem("freeme_token");
    let CURRENT_USER = localStorage.getItem("freeme_user");
    let WISHLIST_IDS = new Set();
    let CURRENT_MODEL = 'internal';
    let CURRENT_FILTER = 'ALL';
    let CURRENT_SORT = 'RELEVANCE';
    let lastSearchData = [];
    let SEARCH_HISTORY = [];
    try { SEARCH_HISTORY = JSON.parse(localStorage.getItem('freeme_history')) || []; } catch (e) { }

    // --- NAVIGATION HELPERS ---
    window.returnToMain = () => {
        document.getElementById('about-modal').classList.add('hidden');
        document.getElementById('auth-modal').classList.add('hidden');
        document.getElementById('sidebar').classList.remove('open');
        document.body.classList.remove('menu-open');

        const fb = document.getElementById('filter-bar');

        if (lastSearchData && lastSearchData.length > 0) {
            renderResults(lastSearchData);
            if (fb) fb.style.display = 'flex';
        } else {
            document.getElementById('results-grid').innerHTML = '';
            if (fb) fb.style.display = 'none';
        }
    };

    window.togglePassword = (btn) => {
        const input = btn.previousElementSibling;
        if (input.type === "password") { input.type = "text"; btn.innerText = "HIDE"; }
        else { input.type = "password"; btn.innerText = "SHOW"; }
    };

    const authModal = document.getElementById("auth-modal");
    const aboutModal = document.getElementById("about-modal");

    window.openAuth = () => { authModal.classList.remove("hidden"); window.switchToLogin(); };
    window.closeAuth = () => authModal.classList.add("hidden");

    window.openAbout = () => { aboutModal.classList.remove("hidden"); sidebar.classList.remove("open"); document.body.classList.remove('menu-open'); };
    window.closeAbout = () => aboutModal.classList.add("hidden");

    window.switchToSignup = () => {
        document.getElementById('login-view').classList.add('hidden');
        document.getElementById('signup-view').classList.remove('hidden');
        document.getElementById('auth-error').innerText = "";
    };
    window.switchToLogin = () => {
        document.getElementById('signup-view').classList.add('hidden');
        document.getElementById('login-view').classList.remove('hidden');
        document.getElementById('auth-error').innerText = "";
    };

    // --- AUTH LOGIC ---
    // --- AUTH LOGIC ---
    // We attach these directly to 'window' so the HTML can definitely see them.
    window.login = async () => {
        console.log("Login button clicked..."); // DEBUG: Check console if this appears

        const btn = document.querySelector('#login-view .primary');
        const originalText = btn.innerText;
        btn.innerText = "CONNECTING..."; // Visual feedback
        btn.disabled = true;

        const u = document.getElementById("auth-username").value;
        const p = document.getElementById("auth-password").value;

        if (!u || !p) {
            document.getElementById('auth-error').innerText = "MISSING CREDENTIALS";
            btn.innerText = originalText;
            btn.disabled = false;
            return;
        }

        try {
            // Use FormData for FastAPI OAuth2 compatibility
            const form = new FormData();
            form.append("username", u);
            form.append("password", p);

            const res = await fetch(`${API_URL}/login`, { method: "POST", body: form });

            if (res.ok) {
                const data = await res.json();
                AUTH_TOKEN = data.access_token;
                CURRENT_USER = u;
                localStorage.setItem("freeme_token", AUTH_TOKEN);
                localStorage.setItem("freeme_user", u);
                window.closeAuth();
                bootSystem();
                alert("LOGIN SUCCESSFUL");
            } else {
                const err = await res.json();
                throw new Error(err.detail || "Invalid Credentials");
            }
        } catch (e) {
            console.error(e);
            document.getElementById('auth-error').innerText = "LOGIN FAILED: " + e.message;
        } finally {
            // Always reset the button
            btn.innerText = originalText;
            btn.disabled = false;
        }
    };

    window.signup = async () => {
        console.log("Signup button clicked...");

        const btn = document.querySelector('#signup-view .primary');
        const originalText = btn.innerText;
        btn.innerText = "REGISTERING...";
        btn.disabled = true;

        const u = document.getElementById("su-username").value;
        const e = document.getElementById("su-email").value;
        const p = document.getElementById("su-password").value;
        const c = document.getElementById("su-confirm").value;

        if (p !== c) {
            document.getElementById('auth-error').innerText = "PASSWORDS DO NOT MATCH";
            btn.innerText = originalText;
            btn.disabled = false;
            return;
        }

        try {
            const res = await fetch(`${API_URL}/signup`, {
                method: "POST",
                headers: { "Content-Type": "application/json" },
                body: JSON.stringify({ username: u, email: e, password: p })
            });

            if (res.ok) {
                window.switchToLogin();
                alert("ACCOUNT CREATED! PLEASE LOGIN.");
            } else {
                const err = await res.json();
                document.getElementById('auth-error').innerText = err.detail || "Signup Failed";
            }
        } catch (error) {
            document.getElementById('auth-error').innerText = "ERROR: " + error.message;
        } finally {
            btn.innerText = originalText;
            btn.disabled = false;
        }
    };

    window.logout = () => {
        localStorage.removeItem('freeme_token');
        localStorage.removeItem('freeme_user');
        localStorage.removeItem('freeme_history');
        location.reload();
    };

    // --- SEARCH LOGIC ---
    function addToHistory(q) {
        SEARCH_HISTORY.unshift({ query: q, timestamp: new Date().toLocaleTimeString(), id: Date.now() });
        if (SEARCH_HISTORY.length > 50) SEARCH_HISTORY.pop();
        localStorage.setItem('freeme_history', JSON.stringify(SEARCH_HISTORY));
    }

    async function performSearch() {
        const query = document.getElementById('search-input').value;
        if (!query) return;
        addToHistory(query);

        const grid = document.getElementById('results-grid');
        grid.innerHTML = `<h2 style="grid-column:1/-1;text-align:center;color:var(--neon-blue);animation:pulse 1s infinite;">NEURAL SCAN IN PROGRESS...</h2>`;
        try {
            const endpoint = AUTH_TOKEN ? "/recommend/personalized" : "/recommend";
            const res = await fetch(`${API_URL}${endpoint}`, { method: "POST", headers: { "Content-Type": "application/json", ...(AUTH_TOKEN && { "Authorization": `Bearer ${AUTH_TOKEN}` }) }, body: JSON.stringify({ text: query, top_k: 12, model: CURRENT_MODEL }) });
            const data = await res.json(); lastSearchData = data; renderResults(data);
        } catch { grid.innerHTML = `<h3 style="text-align:center;grid-column:1/-1;color:red;">CONNECTION ERROR</h3>`; }
    }

    function renderResults(data, isSimilarView = false) {
        const grid = document.getElementById('results-grid'); grid.innerHTML = "";

        if (!isSimilarView && data.length > 0) renderFilters();

        let filtered = [...data];
        if (CURRENT_FILTER !== 'ALL') {
            filtered = filtered.filter(item => {
                const t = (item.type || "").toUpperCase();
                if (CURRENT_FILTER === 'DOC') return t.includes('DOC');
                return t.includes(CURRENT_FILTER);
            });
        }

        // SORTING: Uses the FULL original value
        if (CURRENT_SORT === 'RATING') {
            filtered.sort((a, b) => (parseFloat(b.rating) || 0) - (parseFloat(a.rating) || 0));
        }

        if (!filtered.length) { grid.innerHTML = `<h3 style="text-align:center;grid-column:1/-1;">NO PATTERNS FOUND</h3>`; return; }

        filtered.forEach(item => {
            const card = document.createElement('div'); card.className = 'card';
            let imgUrl = `https://placehold.co/300x450/111/FFF?text=${encodeURIComponent(item.title)}`;
            if (item.image && item.image.length > 5 && !item.image.includes("null")) imgUrl = `https://wsrv.nl/?url=${encodeURIComponent(item.image)}&w=400&output=webp`;
            const isSaved = WISHLIST_IDS.has(item.id);

            let type = (item.type || "MOVIE").toUpperCase();
            let typeClass = "movie";
            if (type.includes("TV")) typeClass = "tv";
            if (type.includes("ANIME")) typeClass = "anime";
            if (type.includes("DOC")) typeClass = "doc";

            // --- FIXED RATING FORMATTING (X.Y) ---
            let ratingVal = parseFloat(item.rating);
            let rating = !isNaN(ratingVal) ? `★ ${ratingVal.toFixed(1)}` : "";
            // -------------------------------------

            const exploreBtn = isSimilarView ? '' : `<button class="similar-btn" onclick="window.findSimilar('${item.id}')">EXPLORE SIMILAR</button>`;

            card.innerHTML = `
                <div class="card-media-wrapper"><img src="${imgUrl}" loading="lazy" onload="this.classList.add('loaded')"><div class="match-bar-track"><span class="match-label-base label-cyan">${item.score || 85}% MATCH</span></div></div>
                <div class="card-content">
                    <div class="badge-row">
                        <span class="type-badge type-${typeClass}">${type}</span>
                        ${rating ? `<span class="rating-badge">${rating}</span>` : ''}
                        <button onclick="window.toggleWishlist(this, '${item.id}')" class="wishlist-btn" style="color:${isSaved ? '#ff0055' : '#888'};">${isSaved ? '♥' : '♡'}</button>
                    </div>
                    <h3>${item.title}</h3><p>${item.description || "No data."}</p>
                    ${exploreBtn}
                </div>`;
            grid.appendChild(card);
        });
    }

    // --- FILTERS ---
    window.setFilter = (type) => {
        CURRENT_FILTER = type;
        document.querySelectorAll('.filter-btn').forEach(b => b.classList.remove('active'));
        document.getElementById(`btn-${type.toLowerCase()}`).classList.add('active');
        renderResults(lastSearchData);
    };

    window.toggleSort = () => {
        CURRENT_SORT = (CURRENT_SORT === 'RELEVANCE') ? 'RATING' : 'RELEVANCE';
        const btn = document.getElementById('btn-sort');
        btn.innerText = `SORT: ${CURRENT_SORT}`;
        btn.classList.toggle('active');
        renderResults(lastSearchData);
    };

    function renderFilters() {
        if (document.getElementById('filter-bar')) return;
        const fb = document.createElement("div"); fb.id = "filter-bar";
        fb.innerHTML = `
            <button onclick="setFilter('ALL')" class="filter-btn active" id="btn-all">ALL</button>
            <button onclick="setFilter('MOVIE')" class="filter-btn" id="btn-movie">MOVIES</button>
            <button onclick="setFilter('TV')" class="filter-btn" id="btn-tv">TV</button>
            <button onclick="setFilter('ANIME')" class="filter-btn" id="btn-anime">ANIME</button>
            <button onclick="setFilter('DOC')" class="filter-btn" id="btn-doc">DOCS</button>
            <div style="width:1px;background:var(--border-color);height:20px;margin:0 10px;"></div>
            <button onclick="toggleSort()" class="filter-btn" id="btn-sort">SORT: RELEVANCE</button>
        `;
        const grid = document.getElementById('results-grid');
        grid.parentNode.insertBefore(fb, grid);
    }

    window.toggleWishlist = async (btn, id) => {
        if (!AUTH_TOKEN) return window.openAuth();
        const action = WISHLIST_IDS.has(id) ? 'remove' : 'add';
        await fetch(`${API_URL}/wishlist/${action}/${id}`, { method: action === 'add' ? 'POST' : 'DELETE', headers: { "Authorization": `Bearer ${AUTH_TOKEN}` } });
        if (action === 'add') { WISHLIST_IDS.add(id); btn.style.color = '#ff0055'; btn.innerText = '♥'; } else { WISHLIST_IDS.delete(id); btn.style.color = '#888'; btn.innerText = '♡'; }
    };

    window.findSimilar = async (id) => {
        const grid = document.getElementById('results-grid');
        const fb = document.getElementById('filter-bar'); if (fb) fb.style.display = 'none';
        grid.innerHTML = `<h2 style="grid-column:1/-1;text-align:center;">VECTOR TRIANGULATION...</h2>`;
        const res = await fetch(`${API_URL}/similar`, { method: "POST", headers: { "Content-Type": "application/json" }, body: JSON.stringify({ id }) });
        const data = await res.json();
        renderResults(data, true);
        const backBtn = document.createElement("button"); backBtn.innerText = "← RETURN TO SEARCH"; backBtn.className = "back-btn";
        backBtn.onclick = () => { if (fb) fb.style.display = 'flex'; renderResults(lastSearchData); };
        grid.insertBefore(backBtn, grid.firstChild);
    };

    window.openWishlist = () => {
        sidebar.classList.remove('open'); document.body.classList.remove('menu-open');
        const grid = document.getElementById('results-grid');

        const fb = document.getElementById('filter-bar'); if (fb) fb.style.display = 'none';

        if (!AUTH_TOKEN) {
            grid.innerHTML = `<h3 style="text-align:center;grid-column:1/-1;color:var(--text-secondary);margin-top:50px;">PLEASE LOGIN TO VIEW WISHLIST</h3>`;
            return;
        }

        grid.innerHTML = `<h2 style="grid-column:1/-1;text-align:center;">LOADING WISHLIST...</h2>`;
        fetch(`${API_URL}/wishlist`, { headers: { "Authorization": `Bearer ${AUTH_TOKEN}` } })
            .then(res => res.json())
            .then(data => {
                if (!data || data.length === 0) {
                    grid.innerHTML = `<h3 style="text-align:center;grid-column:1/-1;margin-top:50px;">YOUR WISHLIST IS EMPTY</h3>`;
                } else {
                    renderResults(data, true);
                }
            })
            .catch(e => { grid.innerHTML = `<h3 style="text-align:center;color:red;">ERROR LOADING WISHLIST</h3>`; });
    };

    window.openHistory = () => {
        sidebar.classList.remove('open'); document.body.classList.remove('menu-open');
        const grid = document.getElementById('results-grid');
        grid.innerHTML = "";
        const fb = document.getElementById('filter-bar'); if (fb) fb.style.display = 'none';
        if (!SEARCH_HISTORY.length) { grid.innerHTML = `<h3 style="text-align:center;grid-column:1/-1;">NO SEARCH HISTORY</h3>`; return; }
        SEARCH_HISTORY.forEach(i => {
            const row = document.createElement('div'); row.className = 'history-row';
            row.style.cssText = "grid-column: 1/-1; background: var(--bg-card);";
            row.innerHTML = `<div class="hist-time">${i.timestamp}</div><div class="hist-query">${i.query}</div><button class="hist-btn">RELOAD</button>`;
            row.querySelector('button').onclick = () => { document.getElementById('search-input').value = i.query; performSearch(); };
            grid.appendChild(row);
        });
    };

    // --- MENU ---
    const menuBtn = document.getElementById('menu-btn'); const sidebar = document.getElementById('sidebar');
    if (menuBtn) menuBtn.addEventListener('click', (e) => { e.stopPropagation(); sidebar.classList.toggle('open'); document.body.classList.toggle('menu-open'); });
    document.addEventListener('click', (e) => { if (sidebar.classList.contains('open') && !sidebar.contains(e.target) && !menuBtn.contains(e.target)) { sidebar.classList.remove('open'); document.body.classList.remove('menu-open'); } });

    document.getElementById('nav-wishlist').addEventListener('click', window.openWishlist);
    document.getElementById('nav-history').addEventListener('click', window.openHistory);
    document.getElementById('nav-about').addEventListener('click', window.openAbout);

    // --- SYSTEM BOOT ---
    async function bootSystem() {
        if (AUTH_TOKEN) {
            document.getElementById('nav-auth').innerText = "LOGOUT"; document.getElementById('nav-auth').onclick = window.logout;
            document.getElementById('hud-user').innerText = CURRENT_USER;
            const res = await fetch(`${API_URL}/wishlist`, { headers: { "Authorization": `Bearer ${AUTH_TOKEN}` } });
            if (res.ok) { const data = await res.json(); WISHLIST_IDS = new Set(data.map(i => i.id)); }
        }
        try { await fetch(`${API_URL}/docs`); document.getElementById('status-indicator').style.background = "#00ff9d"; document.getElementById('status-text').innerText = "ONLINE"; } catch { }
    }

    const toggle = document.getElementById('theme-toggle');
    const isLight = localStorage.getItem('theme') === 'light';
    toggle.checked = isLight;

    toggle.addEventListener('change', (e) => {
        const t = e.target.checked ? 'light' : 'dark';
        if (t === 'light') { document.documentElement.classList.add('light-mode'); }
        else { document.documentElement.classList.remove('light-mode'); }
        localStorage.setItem('theme', t);
    });

    document.getElementById('search-input')?.addEventListener('keydown', (e) => { if (e.key === 'Enter') performSearch(); });
    document.getElementById('search-btn')?.addEventListener('click', performSearch);
    document.getElementById('random-btn')?.addEventListener('click', () => { document.getElementById('search-input').value = ["Cyberpunk Anime", "80s Horror", "Deep Space Sci-Fi", "Noir Mystery"].sort(() => 0.5 - Math.random())[0]; performSearch(); });

    window.toggleModelMenu = () => { document.getElementById('model-menu').classList.toggle('hidden'); setTimeout(() => document.getElementById('model-menu').classList.toggle('open'), 10); };
    window.selectModel = (m, el) => {
        CURRENT_MODEL = m; document.querySelectorAll('.model-option').forEach(o => o.classList.remove('active'));
        el.classList.add('active'); document.getElementById('current-model-name').innerText = el.querySelector('.opt-title').innerText;
        window.toggleModelMenu();
    };

    bootSystem();
});