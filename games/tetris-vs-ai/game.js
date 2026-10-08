(() => {
  'use strict';

  // ---------------------------------------------------------------------------
  // Rules
  // ---------------------------------------------------------------------------

  const COLS = 10;
  const ROWS = 22; // top 2 rows are hidden spawn rows
  const HIDDEN = 2;
  const CELL = 30;

  const COLORS = {
    I: '#00e5f0', O: '#f0d800', T: '#a640f0', S: '#30d050',
    Z: '#f03838', J: '#3a6cf0', L: '#f09a20', G: '#5b6078',
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

  const LOCK_DELAY = 500;
  const MAX_LOCK_RESETS = 15;
  const DAS = 160;
  const ARR = 45;
  const SOFT_DROP_INTERVAL = 35;
  const ATTACK = [0, 0, 1, 2, 4]; // garbage rows sent for 0..4 lines cleared
  const MAX_GARBAGE_PER_LOCK = 8;
  const LEVEL_SECONDS = 40;
  const MAX_LEVEL = 15;

  const rotateCW = (m) => m[0].map((_, i) => m.map((row) => row[i]).reverse());

  const ROTATIONS = {};
  for (const [k, shape] of Object.entries(SHAPES)) {
    const states = [shape];
    for (let i = 1; i < 4; i++) states.push(rotateCW(states[i - 1]));
    ROTATIONS[k] = states;
  }

  // Small seeded PRNG so both boards get the exact same piece sequence.
  function mulberry32(a) {
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  // ---------------------------------------------------------------------------
  // Grid helpers (pure, shared by the boards and the AI search)
  // ---------------------------------------------------------------------------

  const emptyRow = () => Array(COLS).fill(null);
  const emptyGrid = () => Array.from({ length: ROWS }, emptyRow);
  const cells = (p, rot = p.rot) => ROTATIONS[p.type][rot];

  function spawnPiece(type) {
    const size = ROTATIONS[type][0].length;
    return { type, rot: 0, x: Math.floor((COLS - size) / 2), y: type === 'I' ? 0 : 1 };
  }

  function collides(grid, p, dx = 0, dy = 0, rot = p.rot) {
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

  // Rotates p in place using SRS kicks; returns whether it succeeded.
  function rotateOn(grid, p, dir) {
    if (p.type === 'O') return false;
    const from = p.rot;
    const to = (from + (dir > 0 ? 1 : 3)) % 4;
    const kicks = (p.type === 'I' ? KICKS_I : KICKS_JLSTZ)[`${from}>${to}`];
    for (const [kx, ky] of kicks) {
      if (!collides(grid, p, kx, -ky, to)) {
        p.x += kx;
        p.y -= ky;
        p.rot = to;
        return true;
      }
    }
    return false;
  }

  function dropDistance(grid, p) {
    let d = 0;
    while (!collides(grid, p, 0, d + 1)) d++;
    return d;
  }

  // Returns a new grid with p locked in and full rows removed.
  function stamp(grid, p) {
    const g = grid.map((row) => row.slice());
    const m = cells(p);
    let lockOut = true;
    for (let r = 0; r < m.length; r++) {
      for (let c = 0; c < m[r].length; c++) {
        if (!m[r][c]) continue;
        const y = p.y + r;
        if (y >= 0) g[y][p.x + c] = p.type;
        if (y >= HIDDEN) lockOut = false;
      }
    }
    const kept = g.filter((row) => !row.every(Boolean));
    const lines = ROWS - kept.length;
    while (kept.length < ROWS) kept.unshift(emptyRow());
    return { grid: kept, lines, lockOut };
  }

  // ---------------------------------------------------------------------------
  // Board: one player's playfield
  // ---------------------------------------------------------------------------

  class Board {
    constructor(seed, hooks) {
      this.rng = mulberry32(seed);
      this.holeRng = mulberry32(seed ^ 0x9e3779b9);
      this.hooks = hooks;
      this.grid = emptyGrid();
      this.bag = [];
      this.queue = [];
      while (this.queue.length < 5) this.queue.push(this.nextFromBag());
      this.hold = null;
      this.holdUsed = false;
      this.lines = 0;
      this.sent = 0;
      this.pending = 0;
      this.dead = false;
      this.pieceId = 0;
      this.spawn();
    }

    nextFromBag() {
      if (this.bag.length === 0) {
        this.bag = Object.keys(SHAPES);
        for (let i = this.bag.length - 1; i > 0; i--) {
          const j = Math.floor(this.rng() * (i + 1));
          [this.bag[i], this.bag[j]] = [this.bag[j], this.bag[i]];
        }
      }
      return this.bag.pop();
    }

    spawn(type) {
      this.piece = spawnPiece(type ?? this.queue.shift());
      while (this.queue.length < 5) this.queue.push(this.nextFromBag());
      this.pieceId++;
      this.dropTimer = 0;
      this.lockTimer = 0;
      this.lockResets = 0;
      if (collides(this.grid, this.piece)) this.die();
    }

    die() {
      if (this.dead) return;
      this.dead = true;
      this.hooks.onTopOut(this);
    }

    onGround() { return collides(this.grid, this.piece, 0, 1); }

    resetLock() {
      if (this.lockResets < MAX_LOCK_RESETS) {
        this.lockTimer = 0;
        this.lockResets++;
      }
    }

    move(dx) {
      if (this.dead || collides(this.grid, this.piece, dx, 0)) return false;
      this.piece.x += dx;
      this.resetLock();
      return true;
    }

    rotate(dir) {
      if (this.dead || !rotateOn(this.grid, this.piece, dir)) return false;
      this.resetLock();
      return true;
    }

    softDrop() {
      if (this.dead || this.onGround()) return false;
      this.piece.y++;
      this.dropTimer = 0;
      return true;
    }

    hardDrop() {
      if (this.dead) return;
      this.piece.y += dropDistance(this.grid, this.piece);
      this.lock();
    }

    holdPiece() {
      if (this.dead || this.holdUsed) return false;
      const held = this.hold;
      this.hold = this.piece.type;
      this.spawn(held ?? undefined);
      this.holdUsed = true;
      return true;
    }

    lock() {
      const { grid, lines, lockOut } = stamp(this.grid, this.piece);
      this.grid = grid;
      if (lockOut) return this.die();
      this.holdUsed = false;

      if (lines) {
        this.lines += lines;
        let attack = ATTACK[lines];
        // Clearing lines first cancels garbage that is on its way to us.
        const cancel = Math.min(attack, this.pending);
        this.pending -= cancel;
        attack -= cancel;
        if (attack) {
          this.sent += attack;
          this.hooks.onAttack(this, attack);
        }
      } else if (this.pending) {
        const n = Math.min(this.pending, MAX_GARBAGE_PER_LOCK);
        this.pending -= n;
        this.addGarbage(n);
        if (this.dead) return;
      }
      this.spawn();
    }

    addGarbage(n) {
      const hole = Math.floor(this.holeRng() * COLS);
      for (let i = 0; i < n; i++) {
        if (this.grid.shift().some(Boolean)) this.die();
        const row = Array(COLS).fill('G');
        row[hole] = null;
        this.grid.push(row);
      }
    }

    tick(dt, interval) {
      if (this.dead) return;
      if (this.onGround()) {
        this.lockTimer += dt;
        if (this.lockTimer >= LOCK_DELAY) this.lock();
      } else {
        this.dropTimer += dt;
        while (this.dropTimer >= interval && !this.onGround()) {
          this.dropTimer -= interval;
          this.piece.y++;
        }
        if (this.onGround()) this.dropTimer = 0;
      }
    }
  }

  // ---------------------------------------------------------------------------
  // AI: score every reachable placement, then play it out move by move
  // ---------------------------------------------------------------------------

  // Weights from Yiyuan Lee's genetic-algorithm-tuned Tetris heuristic.
  const WEIGHTS = { height: -0.510066, lines: 0.760666, holes: -0.35663, bumpiness: -0.184483 };

  function evaluate(grid, lines) {
    let height = 0, holes = 0, bumpiness = 0, prev = -1;
    for (let x = 0; x < COLS; x++) {
      let y = 0;
      while (y < ROWS && !grid[y][x]) y++;
      const h = ROWS - y;
      for (let yy = y + 1; yy < ROWS; yy++) if (!grid[yy][x]) holes++;
      height += h;
      if (prev >= 0) bumpiness += Math.abs(h - prev);
      prev = h;
    }
    return WEIGHTS.height * height + WEIGHTS.lines * lines
      + WEIGHTS.holes * holes + WEIGHTS.bumpiness * bumpiness;
  }

  // All placements reachable by rotating, then shifting, then hard-dropping.
  function placements(grid, start) {
    const out = [];
    const rotations = start.type === 'O' ? [[]] : [[], ['cw'], ['cw', 'cw'], ['ccw']];
    const land = (p, actions) => {
      const landed = { ...p, y: p.y + dropDistance(grid, p) };
      out.push({ actions: [...actions, 'drop'], ...stamp(grid, landed) });
    };
    for (const rots of rotations) {
      const base = { ...start };
      if (!rots.every((r) => rotateOn(grid, base, r === 'cw' ? 1 : -1))) continue;
      land(base, rots);
      for (const dir of [-1, 1]) {
        const p = { ...base };
        const actions = [...rots];
        while (!collides(grid, p, dir, 0)) {
          p.x += dir;
          actions.push(dir < 0 ? 'left' : 'right');
          land(p, actions);
        }
      }
    }
    return out;
  }

  const score = (c, lines) => (c.lockOut ? -1e9 : evaluate(c.grid, lines));

  function choosePlacement(board, cfg) {
    const candidates = placements(board.grid, board.piece).map((c) => {
      let s = score(c, c.lines);
      if (cfg.lookahead && !c.lockOut) {
        // Judge this placement by the best follow-up with the next piece.
        s = -Infinity;
        for (const c2 of placements(c.grid, spawnPiece(board.queue[0]))) {
          s = Math.max(s, score(c2, c.lines + c2.lines));
        }
      }
      return { ...c, score: s };
    });
    if (candidates.length === 0) return null;
    candidates.sort((a, b) => b.score - a.score);
    if (cfg.mistakes && Math.random() < cfg.mistakes) {
      return candidates[Math.floor(Math.random() * Math.min(cfg.topK, candidates.length))];
    }
    return candidates[0];
  }

  class AIPlayer {
    constructor(board, cfg) {
      this.board = board;
      this.cfg = cfg;
      this.plan = [];
      this.planFor = -1;
      this.timer = 0;
    }

    update(dt) {
      const b = this.board;
      if (b.dead) return;
      if (this.planFor !== b.pieceId) {
        const choice = choosePlacement(b, this.cfg);
        this.plan = choice ? choice.actions.slice() : ['drop'];
        this.planFor = b.pieceId;
        this.timer = -this.cfg.think;
      }
      this.timer += dt;
      while (this.timer >= this.cfg.delay && this.plan.length && this.planFor === b.pieceId) {
        this.timer -= this.cfg.delay;
        const a = this.plan.shift();
        if (a === 'cw') b.rotate(1);
        else if (a === 'ccw') b.rotate(-1);
        else if (a === 'left') b.move(-1);
        else if (a === 'right') b.move(1);
        else b.hardDrop();
      }
    }
  }

  // delay: ms between AI moves; think: ms before the first move of each piece;
  // mistakes: chance of picking one of the topK placements instead of the best.
  const DIFFICULTIES = {
    easy: { label: 'Easy', delay: 260, think: 450, lookahead: false, mistakes: 0.3, topK: 4 },
    normal: { label: 'Normal', delay: 140, think: 250, lookahead: false, mistakes: 0.1, topK: 2 },
    hard: { label: 'Hard', delay: 70, think: 120, lookahead: true, mistakes: 0 },
    insane: { label: 'Insane', delay: 20, think: 0, lookahead: true, mistakes: 0 },
  };

  // ---------------------------------------------------------------------------
  // DOM & rendering
  // ---------------------------------------------------------------------------

  const $ = (id) => document.getElementById(id);
  const view = {
    human: { board: $('h-board').getContext('2d'), next: $('h-next').getContext('2d'), hold: $('h-hold').getContext('2d'),
      lines: $('h-lines'), sent: $('h-sent'), meter: $('h-meter') },
    ai: { board: $('a-board').getContext('2d'), next: $('a-next').getContext('2d'),
      lines: $('a-lines'), sent: $('a-sent'), meter: $('a-meter') },
  };
  const overlay = $('overlay');
  const overlayTitle = $('overlay-title');
  const overlayMsg = $('overlay-msg');
  const startBtn = $('start');

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const clamp = (v) => Math.max(0, Math.min(255, v));
    return `rgb(${clamp((n >> 16) + amt)},${clamp(((n >> 8) & 255) + amt)},${clamp((n & 255) + amt)})`;
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

  function drawBoard(ctx, board) {
    const w = COLS * CELL, h = (ROWS - HIDDEN) * CELL;
    ctx.fillStyle = '#07080f';
    ctx.fillRect(0, 0, w, h);
    ctx.strokeStyle = 'rgba(255,255,255,0.04)';
    ctx.lineWidth = 1;
    for (let x = 1; x < COLS; x++) {
      ctx.beginPath(); ctx.moveTo(x * CELL + 0.5, 0); ctx.lineTo(x * CELL + 0.5, h); ctx.stroke();
    }
    for (let y = 1; y < ROWS - HIDDEN; y++) {
      ctx.beginPath(); ctx.moveTo(0, y * CELL + 0.5); ctx.lineTo(w, y * CELL + 0.5); ctx.stroke();
    }
    if (!board) return;

    for (let y = HIDDEN; y < ROWS; y++) {
      for (let x = 0; x < COLS; x++) {
        const t = board.grid[y][x];
        if (t) drawCell(ctx, x * CELL, (y - HIDDEN) * CELL, CELL, COLORS[t]);
      }
    }

    if (!board.dead) {
      const p = board.piece;
      const m = cells(p);
      const gy = dropDistance(board.grid, p);
      ctx.strokeStyle = COLORS[p.type];
      ctx.lineWidth = 2;
      ctx.globalAlpha = 0.5;
      for (let r = 0; r < m.length; r++) for (let c = 0; c < m[r].length; c++) {
        const y = p.y + gy + r - HIDDEN;
        if (m[r][c] && y >= 0) ctx.strokeRect((p.x + c) * CELL + 2, y * CELL + 2, CELL - 4, CELL - 4);
      }
      ctx.globalAlpha = 1;
      for (let r = 0; r < m.length; r++) for (let c = 0; c < m[r].length; c++) {
        const y = p.y + r - HIDDEN;
        if (m[r][c] && y >= 0) drawCell(ctx, (p.x + c) * CELL, y * CELL, CELL, COLORS[p.type]);
      }
    } else {
      ctx.fillStyle = 'rgba(7, 8, 15, 0.6)';
      ctx.fillRect(0, 0, w, h);
      ctx.fillStyle = '#f03838';
      ctx.font = 'bold 28px system-ui, sans-serif';
      ctx.textAlign = 'center';
      ctx.fillText('TOPPED OUT', w / 2, h / 2);
    }
  }

  function drawMini(c, type, cx, cy, size, alpha = 1) {
    const m = ROTATIONS[type][0];
    const rows = m.map((row, i) => (row.some(Boolean) ? i : -1)).filter((i) => i >= 0);
    const cols = m[0].map((_, i) => (m.some((row) => row[i]) ? i : -1)).filter((i) => i >= 0);
    const ox = cx - (cols.length * size) / 2, oy = cy - (rows.length * size) / 2;
    rows.forEach((r, ri) => cols.forEach((col, ci) => {
      if (m[r][col]) drawCell(c, ox + ci * size, oy + ri * size, size, COLORS[type], alpha);
    }));
  }

  function drawSide(v, board) {
    v.next.clearRect(0, 0, 100, 240);
    if (v.hold) v.hold.clearRect(0, 0, 100, 80);
    if (!board) return;
    board.queue.slice(0, 3).forEach((t, i) => drawMini(v.next, t, 50, 40 + i * 80, i === 0 ? 20 : 16));
    if (v.hold && board.hold) drawMini(v.hold, board.hold, 50, 40, 20, board.holdUsed ? 0.35 : 1);
  }

  function updateHud(v, board) {
    v.lines.textContent = board.lines;
    v.sent.textContent = board.sent;
    v.meter.style.height = `${Math.min(board.pending / (ROWS - HIDDEN), 1) * 100}%`;
  }

  // ---------------------------------------------------------------------------
  // Match flow
  // ---------------------------------------------------------------------------

  let difficulty = localStorage.getItem('tetris-vs-ai-difficulty') || 'normal';
  if (!DIFFICULTIES[difficulty]) difficulty = 'normal';
  const record = JSON.parse(localStorage.getItem('tetris-vs-ai-record') || '{"wins":0,"losses":0}');

  let human = null, ai = null, aiPlayer = null;
  let state = 'idle'; // idle | playing | paused | over
  let elapsed = 0, level = 1, lastTime = 0;

  function renderRecord() { $('record').textContent = `${record.wins}–${record.losses}`; }

  function selectDifficulty(name) {
    difficulty = name;
    localStorage.setItem('tetris-vs-ai-difficulty', name);
    $('ai-level').textContent = DIFFICULTIES[name].label;
    for (const b of $('difficulty').children) b.classList.toggle('selected', b.dataset.level === name);
  }

  function startMatch() {
    const seed = (Math.random() * 2 ** 32) >>> 0;
    const hooks = {
      onAttack: (from, n) => { (from === human ? ai : human).pending += n; },
      onTopOut: (board) => endMatch(board === human ? 'ai' : 'human'),
    };
    human = new Board(seed, hooks);
    ai = new Board(seed, hooks);
    aiPlayer = new AIPlayer(ai, DIFFICULTIES[difficulty]);
    elapsed = 0;
    level = 1;
    resetInput();
    state = 'playing';
    overlay.classList.add('hidden');
    lastTime = performance.now();
  }

  function endMatch(winner) {
    if (state !== 'playing') return;
    state = 'over';
    if (winner === 'human') record.wins++; else record.losses++;
    localStorage.setItem('tetris-vs-ai-record', JSON.stringify(record));
    renderRecord();
    overlayTitle.textContent = winner === 'human' ? 'YOU WIN!' : 'AI WINS';
    overlayMsg.textContent = `You cleared ${human.lines} lines and sent ${human.sent}. `
      + `The AI (${DIFFICULTIES[difficulty].label}) cleared ${ai.lines} and sent ${ai.sent}.`;
    startBtn.textContent = 'Rematch';
    // Let the final board settle on screen before covering it.
    setTimeout(() => { if (state === 'over') overlay.classList.remove('hidden'); }, 900);
  }

  function togglePause() {
    if (state === 'playing') {
      state = 'paused';
      resetInput();
      overlayTitle.textContent = 'PAUSED';
      overlayMsg.textContent = 'Press P or Esc to resume.';
      startBtn.textContent = 'Resume';
      overlay.classList.remove('hidden');
    } else if (state === 'paused') {
      state = 'playing';
      overlay.classList.add('hidden');
      lastTime = performance.now();
    }
  }

  function primaryAction() {
    if (state === 'paused') togglePause(); else if (state !== 'playing') startMatch();
  }

  // ---------------------------------------------------------------------------
  // Human input
  // ---------------------------------------------------------------------------

  const held = { left: false, right: false, soft: false };
  let dasDir = 0, dasTimer = 0, arrTimer = 0, softTimer = 0;

  function resetInput() {
    held.left = held.right = held.soft = false;
    dasDir = 0;
  }

  function pressHorizontal(dir) {
    held[dir < 0 ? 'left' : 'right'] = true;
    dasDir = dir;
    dasTimer = 0;
    arrTimer = 0;
    human.move(dir);
  }

  function releaseHorizontal(dir) {
    held[dir < 0 ? 'left' : 'right'] = false;
    if (dasDir === dir) {
      dasDir = held.left ? -1 : held.right ? 1 : 0;
      dasTimer = 0;
    }
  }

  function denyHold() {
    const side = view.human.hold.canvas.parentElement;
    side.classList.remove('denied');
    void side.offsetWidth; // restart the animation
    side.classList.add('denied');
  }

  function action(name, down) {
    if (state !== 'playing' || human.dead) return;
    switch (name) {
      case 'left': down ? pressHorizontal(-1) : releaseHorizontal(-1); break;
      case 'right': down ? pressHorizontal(1) : releaseHorizontal(1); break;
      case 'soft': held.soft = down; if (down) { human.softDrop(); softTimer = 0; } break;
      case 'hard': if (down) human.hardDrop(); break;
      case 'rotate': if (down) human.rotate(1); break;
      case 'rotateCCW': if (down) human.rotate(-1); break;
      case 'hold': if (down && !human.holdPiece()) denyHold(); break;
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
  window.addEventListener('blur', () => { if (state === 'playing') togglePause(); });

  startBtn.addEventListener('click', primaryAction);
  for (const b of $('difficulty').children) {
    b.addEventListener('click', () => selectDifficulty(b.dataset.level));
  }
  for (const btn of document.querySelectorAll('#touch button')) {
    const a = btn.dataset.action;
    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); action(a, true); });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
      btn.addEventListener(ev, () => action(a, false));
    }
  }

  // ---------------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------------

  function update(dt) {
    elapsed += dt;
    level = Math.min(MAX_LEVEL, 1 + Math.floor(elapsed / 1000 / LEVEL_SECONDS));
    // Tetris guideline gravity curve, shared by both players.
    const interval = Math.pow(0.8 - (level - 1) * 0.007, level - 1) * 1000;

    if (!human.dead) {
      if (dasDir) {
        dasTimer += dt;
        if (dasTimer >= DAS) {
          arrTimer += dt;
          while (arrTimer >= ARR) {
            arrTimer -= ARR;
            if (!human.move(dasDir)) break;
          }
        }
      }
      if (held.soft) {
        softTimer += dt;
        while (softTimer >= SOFT_DROP_INTERVAL) {
          softTimer -= SOFT_DROP_INTERVAL;
          if (!human.softDrop()) break;
        }
      }
      human.tick(dt, interval);
    }
    if (state === 'playing') {
      aiPlayer.update(dt);
      ai.tick(dt, interval);
    }
  }

  function frame(now) {
    const dt = Math.min(now - lastTime, 100);
    lastTime = now;
    if (state === 'playing') update(dt);

    drawBoard(view.human.board, human);
    drawBoard(view.ai.board, ai);
    drawSide(view.human, human);
    drawSide(view.ai, ai);
    if (human) {
      updateHud(view.human, human);
      updateHud(view.ai, ai);
    }
    $('level').textContent = level;
    requestAnimationFrame(frame);
  }

  selectDifficulty(difficulty);
  renderRecord();
  requestAnimationFrame((t) => { lastTime = t; frame(t); });
})();
