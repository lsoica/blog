(() => {
  'use strict';

  const COLS = 10;
  const ROWS = 22; // top 2 rows are hidden spawn rows
  const HIDDEN = 2;
  const CELL = 30;

  const COLORS = {
    I: '#00e5f0', O: '#f0d800', T: '#a640f0', S: '#30d050',
    Z: '#f03838', J: '#3a6cf0', L: '#f09a20',
  };

  const SHAPES = {
    I: [[0,0,0,0],[1,1,1,1],[0,0,0,0],[0,0,0,0]],
    O: [[1,1],[1,1]],
    T: [[0,1,0],[1,1,1],[0,0,0]],
    S: [[0,1,1],[1,1,0],[0,0,0]],
    Z: [[1,1,0],[0,1,1],[0,0,0]],
    J: [[1,0,0],[1,1,1],[0,0,0]],
    L: [[0,0,1],[1,1,1],[0,0,0]],
  };

  // SRS wall kicks, (x, y) with y pointing up as in the guideline.
  const KICKS_JLSTZ = {
    '0>1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
    '1>0': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
    '1>2': [[0,0],[1,0],[1,-1],[0,2],[1,2]],
    '2>1': [[0,0],[-1,0],[-1,1],[0,-2],[-1,-2]],
    '2>3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
    '3>2': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
    '3>0': [[0,0],[-1,0],[-1,-1],[0,2],[-1,2]],
    '0>3': [[0,0],[1,0],[1,1],[0,-2],[1,-2]],
  };
  const KICKS_I = {
    '0>1': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
    '1>0': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
    '1>2': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
    '2>1': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
    '2>3': [[0,0],[2,0],[-1,0],[2,1],[-1,-2]],
    '3>2': [[0,0],[-2,0],[1,0],[-2,-1],[1,2]],
    '3>0': [[0,0],[1,0],[-2,0],[1,-2],[-2,1]],
    '0>3': [[0,0],[-1,0],[2,0],[-1,2],[2,-1]],
  };

  const LINE_SCORES = [0, 100, 300, 500, 800];
  const LOCK_DELAY = 500;
  const MAX_LOCK_RESETS = 15;
  const DAS = 160;
  const ARR = 45;
  const SOFT_DROP_INTERVAL = 35;

  const rotateCW = (m) => m[0].map((_, i) => m.map((row) => row[i]).reverse());

  // Precompute the 4 rotation states of each piece.
  const ROTATIONS = {};
  for (const [k, shape] of Object.entries(SHAPES)) {
    const states = [shape];
    for (let i = 1; i < 4; i++) states.push(rotateCW(states[i - 1]));
    ROTATIONS[k] = states;
  }

  // ---- DOM ----
  const boardCanvas = document.getElementById('board');
  const ctx = boardCanvas.getContext('2d');
  const nextCtx = document.getElementById('next').getContext('2d');
  const holdCtx = document.getElementById('hold').getContext('2d');
  const overlay = document.getElementById('overlay');
  const overlayMsg = document.getElementById('overlay-msg');
  const startBtn = document.getElementById('start');
  const el = {
    score: document.getElementById('score'),
    lines: document.getElementById('lines'),
    level: document.getElementById('level'),
    best: document.getElementById('best'),
  };

  // ---- State ----
  let grid, piece, queue, bag, hold, holdUsed;
  let score, lines, level, best = Number(localStorage.getItem('tetris-best') || 0);
  let state = 'idle'; // idle | playing | paused | over
  let dropTimer = 0, lockTimer = 0, lockResets = 0, lastTime = 0;
  el.best.textContent = best;

  const emptyGrid = () => Array.from({ length: ROWS }, () => Array(COLS).fill(null));

  function nextFromBag() {
    if (!bag || bag.length === 0) {
      bag = Object.keys(SHAPES);
      for (let i = bag.length - 1; i > 0; i--) {
        const j = Math.floor(Math.random() * (i + 1));
        [bag[i], bag[j]] = [bag[j], bag[i]];
      }
    }
    return bag.pop();
  }

  function makePiece(type) {
    const size = ROTATIONS[type][0].length;
    return { type, rot: 0, x: Math.floor((COLS - size) / 2), y: type === 'I' ? 0 : 1 };
  }

  const cells = (p, rot = p.rot) => ROTATIONS[p.type][rot];

  function collides(p, dx = 0, dy = 0, rot = p.rot) {
    const m = cells(p, rot);
    for (let r = 0; r < m.length; r++) {
      for (let c = 0; c < m[r].length; c++) {
        if (!m[r][c]) continue;
        const x = p.x + c + dx;
        const y = p.y + r + dy;
        if (x < 0 || x >= COLS || y >= ROWS) return true;
        if (y >= 0 && grid[y][x]) return true;
      }
    }
    return false;
  }

  function spawn(type) {
    piece = makePiece(type ?? queue.shift());
    while (queue.length < 5) queue.push(nextFromBag());
    dropTimer = 0;
    lockTimer = 0;
    lockResets = 0;
    if (collides(piece)) gameOver();
  }

  function onGround() { return collides(piece, 0, 1); }

  function resetLockIfGrounded() {
    if (lockResets < MAX_LOCK_RESETS) {
      lockTimer = 0;
      lockResets++;
    }
  }

  function move(dx) {
    if (!collides(piece, dx, 0)) {
      piece.x += dx;
      resetLockIfGrounded();
      return true;
    }
    return false;
  }

  function rotate(dir) {
    if (piece.type === 'O') return;
    const from = piece.rot;
    const to = (from + (dir > 0 ? 1 : 3)) % 4;
    const kicks = (piece.type === 'I' ? KICKS_I : KICKS_JLSTZ)[`${from}>${to}`];
    for (const [kx, ky] of kicks) {
      if (!collides(piece, kx, -ky, to)) {
        piece.x += kx;
        piece.y -= ky;
        piece.rot = to;
        resetLockIfGrounded();
        return;
      }
    }
  }

  function softDrop() {
    if (!collides(piece, 0, 1)) {
      piece.y++;
      score += 1;
      dropTimer = 0;
      updateStats();
      return true;
    }
    return false;
  }

  function hardDrop() {
    let dist = 0;
    while (!collides(piece, 0, 1)) { piece.y++; dist++; }
    score += dist * 2;
    lock();
  }

  function holdPiece() {
    if (holdUsed) {
      // Only one hold per piece; shake the Hold box so the press isn't silently ignored.
      const box = holdCtx.canvas.parentElement;
      box.classList.remove('denied');
      void box.offsetWidth; // restart the animation
      box.classList.add('denied');
      return;
    }
    const current = piece.type;
    if (hold) spawn(hold); else spawn();
    hold = current;
    holdUsed = true;
  }

  function lock() {
    const m = cells(piece);
    let aboveVisible = true;
    for (let r = 0; r < m.length; r++) {
      for (let c = 0; c < m[r].length; c++) {
        if (!m[r][c]) continue;
        const y = piece.y + r;
        if (y >= 0) grid[y][piece.x + c] = piece.type;
        if (y >= HIDDEN) aboveVisible = false;
      }
    }
    // Lock out: piece locked entirely in the hidden zone.
    if (aboveVisible) return gameOver();

    let cleared = 0;
    for (let y = ROWS - 1; y >= 0; y--) {
      if (grid[y].every(Boolean)) {
        grid.splice(y, 1);
        grid.unshift(Array(COLS).fill(null));
        cleared++;
        y++;
      }
    }
    if (cleared) {
      score += LINE_SCORES[cleared] * level;
      lines += cleared;
      level = Math.floor(lines / 10) + 1;
    }
    updateStats();
    holdUsed = false;
    spawn();
  }

  function gravityInterval() {
    // Tetris guideline gravity curve.
    return Math.pow(0.8 - (level - 1) * 0.007, level - 1) * 1000;
  }

  function updateStats() {
    el.score.textContent = score;
    el.lines.textContent = lines;
    el.level.textContent = level;
    if (score > best) {
      best = score;
      el.best.textContent = best;
      localStorage.setItem('tetris-best', String(best));
    }
  }

  // ---- Game flow ----
  function start() {
    grid = emptyGrid();
    bag = [];
    queue = [];
    while (queue.length < 5) queue.push(nextFromBag());
    hold = null;
    holdUsed = false;
    score = 0; lines = 0; level = 1;
    updateStats();
    spawn();
    state = 'playing';
    overlay.classList.add('hidden');
    lastTime = performance.now();
  }

  function gameOver() {
    state = 'over';
    overlay.querySelector('h1').textContent = 'GAME OVER';
    overlayMsg.textContent = `Score ${score} · Press Enter or tap to retry`;
    startBtn.textContent = 'Play again';
    overlay.classList.remove('hidden');
  }

  function togglePause() {
    if (state === 'playing') {
      state = 'paused';
      overlay.querySelector('h1').textContent = 'PAUSED';
      overlayMsg.textContent = 'Press P or Esc to resume';
      startBtn.textContent = 'Resume';
      overlay.classList.remove('hidden');
    } else if (state === 'paused') {
      state = 'playing';
      overlay.classList.add('hidden');
      lastTime = performance.now();
    }
  }

  function primaryAction() {
    if (state === 'paused') togglePause(); else if (state !== 'playing') start();
  }

  // ---- Input ----
  const held = { left: false, right: false, soft: false };
  let dasDir = 0, dasTimer = 0, arrTimer = 0, softTimer = 0;

  function pressHorizontal(dir) {
    held[dir < 0 ? 'left' : 'right'] = true;
    dasDir = dir;
    dasTimer = 0;
    arrTimer = 0;
    move(dir);
  }
  function releaseHorizontal(dir) {
    held[dir < 0 ? 'left' : 'right'] = false;
    if (dasDir === dir) {
      dasDir = held.left ? -1 : held.right ? 1 : 0;
      dasTimer = 0;
    }
  }

  function action(name, down) {
    if (state !== 'playing') return;
    switch (name) {
      case 'left': down ? pressHorizontal(-1) : releaseHorizontal(-1); break;
      case 'right': down ? pressHorizontal(1) : releaseHorizontal(1); break;
      case 'soft': held.soft = down; if (down) { softDrop(); softTimer = 0; } break;
      case 'hard': if (down) hardDrop(); break;
      case 'rotate': if (down) rotate(1); break;
      case 'rotateCCW': if (down) rotate(-1); break;
      case 'hold': if (down) holdPiece(); break;
    }
  }

  // Match on the character typed first (works on any keyboard layout),
  // then fall back to the physical key position.
  const KEYS = {
    arrowleft: 'left', arrowright: 'right', arrowdown: 'soft', arrowup: 'rotate',
    x: 'rotate', z: 'rotateCCW', ' ': 'hard', c: 'hold', shift: 'hold',
    enter: 'start', p: 'pause', escape: 'pause',
  };
  const CODES = {
    ArrowLeft: 'left', ArrowRight: 'right', ArrowDown: 'soft',
    ArrowUp: 'rotate', KeyX: 'rotate', KeyZ: 'rotateCCW',
    Space: 'hard', KeyC: 'hold', ShiftLeft: 'hold', ShiftRight: 'hold',
    Enter: 'start', KeyP: 'pause', Escape: 'pause',
  };
  const keyAction = (e) => KEYS[(e.key || '').toLowerCase()] ?? CODES[e.code];

  document.addEventListener('keydown', (e) => {
    const a = keyAction(e);
    if (a === 'start') { primaryAction(); e.preventDefault(); return; }
    if (a === 'pause') { togglePause(); e.preventDefault(); return; }
    if (!a) return;
    e.preventDefault();
    if (e.repeat) return; // we handle auto-repeat ourselves
    action(a, true);
  });
  document.addEventListener('keyup', (e) => {
    const a = keyAction(e);
    if (a) action(a, false);
  });
  window.addEventListener('blur', () => {
    held.left = held.right = held.soft = false;
    dasDir = 0;
    if (state === 'playing') togglePause();
  });

  startBtn.addEventListener('click', (e) => { e.stopPropagation(); primaryAction(); });
  overlay.addEventListener('click', primaryAction);

  for (const btn of document.querySelectorAll('#touch button')) {
    const a = btn.dataset.action;
    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); action(a, true); });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
      btn.addEventListener(ev, () => action(a, false));
    }
  }

  // ---- Loop ----
  function update(dt) {
    // Horizontal auto-repeat
    if (dasDir) {
      dasTimer += dt;
      if (dasTimer >= DAS) {
        arrTimer += dt;
        while (arrTimer >= ARR) {
          arrTimer -= ARR;
          if (!move(dasDir)) break;
        }
      }
    }

    // Soft drop repeat
    if (held.soft) {
      softTimer += dt;
      while (softTimer >= SOFT_DROP_INTERVAL) {
        softTimer -= SOFT_DROP_INTERVAL;
        if (!softDrop()) break;
      }
    }

    // Gravity / lock
    if (onGround()) {
      lockTimer += dt;
      if (lockTimer >= LOCK_DELAY) lock();
    } else {
      dropTimer += dt;
      const interval = gravityInterval();
      while (dropTimer >= interval && !onGround()) {
        dropTimer -= interval;
        piece.y++;
      }
      if (onGround()) dropTimer = 0;
    }
  }

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const clamp = (v) => Math.max(0, Math.min(255, v));
    const r = clamp((n >> 16) + amt), g = clamp(((n >> 8) & 255) + amt), b = clamp((n & 255) + amt);
    return `rgb(${r},${g},${b})`;
  }

  function drawCell(c, x, y, size, color, alpha = 1) {
    c.globalAlpha = alpha;
    c.fillStyle = color;
    c.fillRect(x, y, size, size);
    c.fillStyle = shade(color, 60);
    c.fillRect(x, y, size, size * 0.15);
    c.fillRect(x, y, size * 0.15, size);
    c.fillStyle = shade(color, -70);
    c.fillRect(x, y + size * 0.85, size, size * 0.15);
    c.fillRect(x + size * 0.85, y, size * 0.15, size);
    c.globalAlpha = 1;
  }

  function drawBoard() {
    ctx.fillStyle = '#07080f';
    ctx.fillRect(0, 0, boardCanvas.width, boardCanvas.height);

    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let x = 1; x < COLS; x++) {
      ctx.beginPath(); ctx.moveTo(x * CELL + 0.5, 0); ctx.lineTo(x * CELL + 0.5, boardCanvas.height); ctx.stroke();
    }
    for (let y = 1; y < ROWS - HIDDEN; y++) {
      ctx.beginPath(); ctx.moveTo(0, y * CELL + 0.5); ctx.lineTo(boardCanvas.width, y * CELL + 0.5); ctx.stroke();
    }

    if (!grid) return;

    for (let y = HIDDEN; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const t = grid[y][x];
        if (t) drawCell(ctx, x * CELL, (y - HIDDEN) * CELL, CELL, COLORS[t]);
      }
    }

    if (!piece || state === 'over') return;
    const m = cells(piece);

    // Ghost
    let gy = 0;
    while (!collides(piece, 0, gy + 1)) gy++;
    ctx.strokeStyle = COLORS[piece.type];
    ctx.lineWidth = 2;
    ctx.globalAlpha = 0.5;
    for (let r = 0; r < m.length; r++) for (let c = 0; c < m[r].length; c++) {
      if (!m[r][c]) continue;
      const y = piece.y + gy + r - HIDDEN;
      if (y < 0) continue;
      ctx.strokeRect((piece.x + c) * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4);
    }
    ctx.globalAlpha = 1;

    // Active piece
    for (let r = 0; r < m.length; r++) for (let c = 0; c < m[r].length; c++) {
      if (!m[r][c]) continue;
      const y = piece.y + r - HIDDEN;
      if (y < 0) continue;
      drawCell(ctx, (piece.x + c) * CELL, y * CELL, CELL, COLORS[piece.type]);
    }
  }

  function drawMini(c, type, cx, cy, size, alpha = 1) {
    const m = ROTATIONS[type][0];
    // Trim empty rows/cols so the piece is centered.
    const rows = m.map((row, i) => (row.some(Boolean) ? i : -1)).filter((i) => i >= 0);
    const colsUsed = m[0].map((_, i) => (m.some((row) => row[i]) ? i : -1)).filter((i) => i >= 0);
    const w = colsUsed.length * size, h = rows.length * size;
    const ox = cx - w / 2, oy = cy - h / 2;
    rows.forEach((r, ri) => colsUsed.forEach((col, ci) => {
      if (m[r][col]) drawCell(c, ox + ci * size, oy + ri * size, size, COLORS[type], alpha);
    }));
  }

  function drawSide() {
    nextCtx.clearRect(0, 0, 120, 360);
    holdCtx.clearRect(0, 0, 120, 120);
    if (!queue) return;
    queue.slice(0, 3).forEach((t, i) => drawMini(nextCtx, t, 60, 60 + i * 120, i === 0 ? 24 : 20));
    if (hold) drawMini(holdCtx, hold, 60, 60, 24, holdUsed ? 0.35 : 1);
  }

  function frame(now) {
    const dt = Math.min(now - lastTime, 100);
    lastTime = now;
    if (state === 'playing') update(dt);
    drawBoard();
    drawSide();
    requestAnimationFrame(frame);
  }

  requestAnimationFrame((t) => { lastTime = t; frame(t); });
})();
