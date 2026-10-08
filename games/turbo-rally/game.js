(() => {
  'use strict';

  // ===========================================================================
  // Turbo Rally: a pseudo-3D stage racer in the spirit of the 16-bit classics.
  // Tracks, physics and rendering live in engine.js (window.TurboEngine), shared
  // with the AI Academy; this file is the game: menus, clock, rivals and sound.
  // ===========================================================================

  const {
    SEG, DRAW_DISTANCE, PLAYER_Z, STEP, MAX_SPEED, TOP_KMH, RIVAL_COUNT, CAR_W,
    mulberry32, clamp, lerp, overlap,
    THEMES, STAGES, RIVAL_SPRITES, buildTrack,
    stepCar, HIT_SCENERY, HIT_CAR, autopilot, gearbox, resetWeather, updateWeather,
  } = window.TurboEngine;

  // ---------------------------------------------------------------------------
  // Sound: a synthesised engine, crash noise and beeps
  // ---------------------------------------------------------------------------

  const sound = {
    ctx: null,
    muted: localStorage.getItem('turbo-rally-muted') === '1',

    init() {
      if (this.ctx) {
        if (this.ctx.state === 'suspended') this.ctx.resume();
        return;
      }
      const AC = window.AudioContext || window.webkitAudioContext;
      if (!AC) return;
      const ctx = this.ctx = new AC();
      this.master = ctx.createGain();
      this.master.gain.value = this.muted ? 0 : 0.6;
      this.master.connect(ctx.destination);

      this.engineGain = ctx.createGain();
      this.engineGain.gain.value = 0;
      const filter = ctx.createBiquadFilter();
      filter.type = 'lowpass';
      filter.frequency.value = 900;
      this.osc1 = ctx.createOscillator();
      this.osc1.type = 'sawtooth';
      this.osc2 = ctx.createOscillator();
      this.osc2.type = 'square';
      this.osc1.connect(filter);
      this.osc2.connect(filter);
      filter.connect(this.engineGain);
      this.engineGain.connect(this.master);
      this.osc1.start();
      this.osc2.start();

      const len = ctx.sampleRate * 0.4;
      this.noise = ctx.createBuffer(1, len, ctx.sampleRate);
      const data = this.noise.getChannelData(0);
      for (let i = 0; i < len; i++) data[i] = (Math.random() * 2 - 1) * (1 - i / len);
    },

    engine(rpm, throttle, on) {
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const f = 55 + rpm * 150;
      this.osc1.frequency.setTargetAtTime(f, t, 0.03);
      this.osc2.frequency.setTargetAtTime(f * 0.501, t, 0.03);
      this.engineGain.gain.setTargetAtTime(on ? 0.05 + throttle * 0.06 : 0, t, 0.08);
    },

    crash() {
      if (!this.ctx) return;
      const src = this.ctx.createBufferSource();
      src.buffer = this.noise;
      const g = this.ctx.createGain();
      g.gain.value = 0.5;
      src.connect(g);
      g.connect(this.master);
      src.start();
    },

    beep(freq, duration = 0.12, type = 'square') {
      if (!this.ctx) return;
      const t = this.ctx.currentTime;
      const o = this.ctx.createOscillator();
      const g = this.ctx.createGain();
      o.type = type;
      o.frequency.value = freq;
      g.gain.setValueAtTime(0.12, t);
      g.gain.exponentialRampToValueAtTime(0.001, t + duration);
      o.connect(g);
      g.connect(this.master);
      o.start(t);
      o.stop(t + duration);
    },

    toggleMute() {
      this.muted = !this.muted;
      localStorage.setItem('turbo-rally-muted', this.muted ? '1' : '0');
      if (this.master) this.master.gain.value = this.muted ? 0 : 0.6;
    },
  };

  // ---------------------------------------------------------------------------
  // Game state
  // ---------------------------------------------------------------------------

  const canvas = document.getElementById('canvas');
  window.TurboEngine.attach(canvas);
  const $ = (id) => document.getElementById(id);

  const game = {
    state: 'attract', // attract | countdown | racing | timeup | finished | paused | menu
    stageIndex: 0,
    stage: null,
    theme: null,
    track: null,
    position: 0,
    playerX: 0,
    speed: 0,
    steer: 0,
    throttle: 0,
    timeLeft: 0,
    nextCheckpoint: 0,
    countdown: 0,
    score: 0,
    stageStartScore: 0,
    rivals: [],
    skyOffset: 0,
    nearOffset: 0,
    shake: 0,
    crashCooldown: 0,
    stateTimer: 0,
    autopilot: false,
    pausedFrom: null,
  };

  const progress = {
    unlocked: Number(localStorage.getItem('turbo-rally-unlocked') || 1),
    best: Number(localStorage.getItem('turbo-rally-best') || 0),
  };

  function loadStage(index, { attract = false } = {}) {
    const stage = STAGES[index];
    game.stageIndex = index;
    game.stage = stage;
    game.theme = THEMES[stage.theme];
    game.themeName = stage.theme;
    game.track = buildTrack(stage);
    game.position = 0;
    game.playerX = 0;
    game.speed = 0;
    game.steer = 0;
    game.throttle = 0;
    game.nextCheckpoint = 0;
    game.timeLeft = game.track.budgets[0];
    game.skyOffset = 0;
    game.nearOffset = 0;
    game.shake = 0;
    game.hits = { scenery: 0, cars: 0 };
    game.autopilot = attract;
    game.rivals = makeRivals(game.track, stage, mulberry32(stage.seed + 99));
    resetWeather(game.theme);
    buildProgressTicks();
    $('stage-label').textContent = `STAGE ${index + 1} · ${stage.name.toUpperCase()}`;
  }

  function makeRivals(track, stage, rng) {
    const rivals = [];
    for (let k = 0; k < RIVAL_COUNT; k++) {
      const row = Math.floor(k / 2);
      const z = PLAYER_Z + (row + 1) * SEG * 5;
      const car = {
        z,
        offset: k % 2 ? -0.5 : 0.5,
        speed: MAX_SPEED * lerp(stage.rivals[0], stage.rivals[1], rng()),
        sprite: RIVAL_SPRITES[k % RIVAL_SPRITES.length],
        percent: 0,
        finished: false,
      };
      track.find(z).cars.push(car);
      rivals.push(car);
    }
    return rivals;
  }

  function startStage(index) {
    sound.init();
    loadStage(index);
    game.stageStartScore = game.score;
    game.state = 'countdown';
    game.countdown = 3.5;
    game.lastCount = null;
    game.overlayShown = false;
    game.dryRun = false;
    hideOverlay();
    $('hud').classList.remove('hidden');
    showMessage('', 0);
  }

  function startAttract() {
    loadStage(0, { attract: true });
    game.state = 'attract';
    $('hud').classList.add('hidden');
  }

  // ---------------------------------------------------------------------------
  // Input
  // ---------------------------------------------------------------------------

  const input = { left: false, right: false, up: false, down: false };

  const KEYS = {
    arrowleft: 'left', a: 'left', arrowright: 'right', d: 'right',
    arrowup: 'up', w: 'up', arrowdown: 'down', s: 'down',
    p: 'pause', escape: 'pause', m: 'mute', enter: 'enter', f: 'fullscreen',
  };
  const CODES = {
    ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right',
    ArrowUp: 'up', KeyW: 'up', ArrowDown: 'down', KeyS: 'down',
    KeyP: 'pause', Escape: 'pause', KeyM: 'mute', Enter: 'enter', KeyF: 'fullscreen',
  };
  const keyAction = (e) => KEYS[(e.key || '').toLowerCase()] ?? CODES[e.code];

  document.addEventListener('keydown', (e) => {
    const a = keyAction(e);
    if (!a) return;
    e.preventDefault();
    if (a in input) { input[a] = true; return; }
    if (e.repeat) return;
    if (a === 'pause') togglePause();
    else if (a === 'mute') sound.toggleMute();
    else if (a === 'fullscreen') toggleFullscreen();
    else if (a === 'enter') overlayPrimary?.();
  });
  document.addEventListener('keyup', (e) => {
    const a = keyAction(e);
    if (a in input) input[a] = false;
  });
  window.addEventListener('blur', () => {
    for (const k in input) input[k] = false;
    if (game.state === 'racing' || game.state === 'countdown') togglePause();
  });

  for (const btn of document.querySelectorAll('#touch button')) {
    const k = btn.dataset.key;
    btn.addEventListener('pointerdown', (e) => { e.preventDefault(); sound.init(); input[k] = true; });
    for (const ev of ['pointerup', 'pointerleave', 'pointercancel']) {
      btn.addEventListener(ev, () => { input[k] = false; });
    }
  }

  // ---------------------------------------------------------------------------
  // Update
  // ---------------------------------------------------------------------------

  function update(dt) {
    const t = game.track;
    const th = game.theme;
    const playerSeg = t.find(game.position + PLAYER_Z);
    const speedPct = game.speed / MAX_SPEED;
    const dx = dt * 2 * speedPct;
    const startPosition = game.position;
    const driving = game.state === 'racing' || game.state === 'attract';

    let controls = { left: false, right: false, up: false, down: false };
    if (game.autopilot || game.state === 'finished') controls = autopilot(game, game.track, game.theme);
    else if (driving) controls = input;
    if (game.state === 'finished') controls.up = false;

    if (game.state === 'countdown') {
      game.countdown -= dt;
      const shown = Math.ceil(game.countdown - 0.5);
      if (shown !== game.lastCount) {
        game.lastCount = shown;
        if (shown > 0) { showMessage(String(shown)); sound.beep(440); }
        else { showMessage('GO!', 1); sound.beep(880, 0.3); }
      }
      if (game.countdown <= 0.5) game.state = 'racing';
      sound.engine(input.up ? 0.7 : 0.15, input.up ? 1 : 0, true);
      updateWeather(dt, game.theme, playerSeg.curve, 0, game.steer);
      return;
    }

    updateRivals(dt, playerSeg);

    const hits = stepCar(game, controls, t, th, dt);
    if (hits & HIT_SCENERY) {
      game.hits.scenery++;
      crash();
    }
    if (hits & HIT_CAR) {
      game.hits.cars++;
      crash(0.15);
    }

    game.crashCooldown -= dt;
    game.shake = Math.max(0, game.shake - dt);

    const travelled = (game.position - startPosition) / SEG;
    game.skyOffset = (game.skyOffset + 0.0012 * playerSeg.curve * travelled + 1) % 1;
    game.nearOffset = (game.nearOffset + 0.0024 * playerSeg.curve * travelled + 1) % 1;

    updateWeather(dt, th, playerSeg.curve, speedPct, game.steer);
    updateRace(dt);

    const { rpm } = gearbox(game.speed);
    sound.engine(rpm, game.throttle, game.state !== 'attract' && game.state !== 'menu');
  }

  function crash(strength = 0.3) {
    game.shake = Math.max(game.shake, strength);
    if (game.crashCooldown <= 0) {
      game.crashCooldown = 0.4;
      sound.crash();
    }
  }

  function updateRace(dt) {
    const t = game.track;
    const z = game.position + PLAYER_Z;

    if (game.state === 'attract') {
      if (z >= t.finish * SEG) startAttract();
      return;
    }
    if (game.state === 'racing') {
      game.timeLeft -= dt;
      if (game.timeLeft <= 0) {
        game.timeLeft = 0;
        game.state = 'timeup';
        game.stateTimer = 0;
        game.overlayShown = false;
        showMessage('TIME UP', 3);
        sound.beep(220, 0.6, 'sawtooth');
      }
    }
    if (game.state === 'racing' && z >= t.checkpoints[game.nextCheckpoint] * SEG) {
      game.nextCheckpoint++;
      if (game.nextCheckpoint >= t.checkpoints.length) {
        finishStage();
      } else {
        const bonus = t.budgets[game.nextCheckpoint];
        game.timeLeft += bonus;
        showMessage(`CHECKPOINT<small>+${bonus} SECONDS</small>`, 2);
        sound.beep(660, 0.1);
        setTimeout(() => sound.beep(990, 0.15), 110);
      }
    }
    if (game.state === 'timeup' || game.state === 'finished') {
      game.stateTimer += dt;
      if (game.state === 'timeup' && game.stateTimer > 2.5 && !game.overlayShown) {
        game.overlayShown = true;
        showTimeUp();
      }
      if (game.state === 'finished' && game.stateTimer > 2.5 && !game.overlayShown) {
        game.overlayShown = true;
        showStageClear();
      }
    }
  }

  function position() {
    const z = game.position + PLAYER_Z;
    return 1 + game.rivals.filter((c) => c.finished || c.z > z).length;
  }

  function finishStage() {
    game.state = 'finished';
    game.stateTimer = 0;
    game.overlayShown = false;
    const pos = position();
    const timeBonus = Math.round(game.timeLeft * 100);
    const posBonus = Math.max(0, 21 - pos) * 200;
    game.result = { pos, timeLeft: game.timeLeft, points: 1000 + timeBonus + posBonus };
    game.score += game.result.points;
    if (!game.dryRun) {
      progress.unlocked = Math.max(progress.unlocked, Math.min(STAGES.length, game.stageIndex + 2));
      localStorage.setItem('turbo-rally-unlocked', String(progress.unlocked));
      saveBest();
    }
    showMessage(`FINISH<small>POSITION ${pos} / ${RIVAL_COUNT + 1}</small>`, 3);
    sound.beep(523, 0.15);
    setTimeout(() => sound.beep(659, 0.15), 150);
    setTimeout(() => sound.beep(784, 0.35), 300);
  }

  function saveBest() {
    if (!game.dryRun && game.score > progress.best) {
      progress.best = game.score;
      localStorage.setItem('turbo-rally-best', String(progress.best));
    }
  }

  function updateRivals(dt, playerSeg) {
    const t = game.track;
    const segs = t.segments;
    const finishZ = (t.finish + 3) * SEG;
    for (const car of game.rivals) {
      if (car.finished) continue;
      const oldSeg = t.find(car.z);
      car.offset = clamp(car.offset + steerRival(car, oldSeg, playerSeg, segs), -0.9, 0.9);
      car.z += dt * car.speed;
      car.percent = (car.z % SEG) / SEG;
      const newSeg = t.find(car.z);
      if (car.z >= finishZ) {
        car.finished = true;
        oldSeg.cars.splice(oldSeg.cars.indexOf(car), 1);
        continue;
      }
      if (oldSeg !== newSeg) {
        oldSeg.cars.splice(oldSeg.cars.indexOf(car), 1);
        newSeg.cars.push(car);
      }
    }
  }

  // Rivals swerve around slower cars (and the player) a few segments ahead.
  function steerRival(car, carSeg, playerSeg, segs) {
    if (Math.abs(carSeg.index - playerSeg.index) > DRAW_DISTANCE) return 0;
    for (let i = 1; i < 20; i++) {
      const seg = segs[Math.min(segs.length - 1, carSeg.index + i)];
      if (seg === playerSeg && car.speed > game.speed && overlap(game.playerX, CAR_W, car.offset, CAR_W, 1.2)) {
        const dir = game.playerX > 0.5 ? -1 : game.playerX < -0.5 ? 1 : car.offset > game.playerX ? 1 : -1;
        return (dir / i) * (car.speed - game.speed) / MAX_SPEED;
      }
      for (const other of seg.cars) {
        if (other !== car && car.speed > other.speed && overlap(car.offset, CAR_W, other.offset, CAR_W, 1.2)) {
          const dir = other.offset > 0.5 ? -1 : other.offset < -0.5 ? 1 : car.offset > other.offset ? 1 : -1;
          return (dir / i) * (car.speed - other.speed) / MAX_SPEED;
        }
      }
    }
    if (car.offset < -0.8) return 0.02;
    if (car.offset > 0.8) return -0.02;
    return 0;
  }

  // ---------------------------------------------------------------------------
  // HUD & menus
  // ---------------------------------------------------------------------------

  const hud = {
    time: $('time'), pos: $('pos'), speed: $('speed'), gear: $('gear'),
    revs: $('revs'), score: $('score'), fill: $('progress-fill'), timeBox: $('time').parentElement,
  };
  const lastHud = {};
  function setText(key, value) {
    if (lastHud[key] !== value) {
      lastHud[key] = value;
      hud[key].textContent = value;
    }
  }

  function updateHud() {
    if (game.state === 'attract') return;
    const { gear, rpm } = gearbox(game.speed);
    setText('time', String(Math.ceil(game.timeLeft)));
    hud.timeBox.classList.toggle('low', game.state === 'racing' && game.timeLeft <= 10);
    setText('pos', `${position()}/${RIVAL_COUNT + 1}`);
    setText('speed', String(Math.round((game.speed / MAX_SPEED) * TOP_KMH)));
    setText('gear', String(gear));
    setText('score', String(game.score));
    hud.revs.style.width = `${Math.round(rpm * 100)}%`;
    hud.fill.style.width = `${clamp((game.position + PLAYER_Z) / (game.track.finish * SEG), 0, 1) * 100}%`;
  }

  function buildProgressTicks() {
    const bar = $('progress');
    for (const t of bar.querySelectorAll('.tick')) t.remove();
    for (const idx of game.track.checkpoints) {
      const tick = document.createElement('div');
      tick.className = 'tick';
      tick.style.left = `${(idx / game.track.finish) * 100}%`;
      bar.appendChild(tick);
    }
  }

  let messageTimer = null;
  function showMessage(html, seconds = 0) {
    const el = $('message');
    el.innerHTML = html;
    clearTimeout(messageTimer);
    if (seconds) messageTimer = setTimeout(() => { el.innerHTML = ''; }, seconds * 1000);
  }

  const overlay = $('overlay');
  let overlayPrimary = null;

  function showOverlay({ title, body = '', buttons = [] }) {
    overlay.innerHTML = '';
    const h = document.createElement('h1');
    h.textContent = title;
    overlay.appendChild(h);
    if (body) {
      const div = document.createElement('div');
      div.innerHTML = body;
      while (div.firstChild) overlay.appendChild(div.firstChild);
    }
    overlayPrimary = null;
    for (const group of buttons) {
      const row = document.createElement('div');
      row.className = 'row';
      for (const b of group) {
        const el = document.createElement('button');
        el.textContent = b.label;
        if (b.primary) { el.className = 'primary'; overlayPrimary = b.action; }
        if (b.disabled) el.disabled = true;
        el.addEventListener('click', () => b.action());
        row.appendChild(el);
      }
      overlay.appendChild(row);
    }
    const hint = document.createElement('p');
    hint.className = 'hint';
    hint.textContent = 'Enter to continue · M toggles sound';
    overlay.appendChild(hint);
    overlay.classList.remove('hidden');
  }

  function hideOverlay() {
    overlay.classList.add('hidden');
    overlayPrimary = null;
  }

  function showTitle() {
    if (game.state !== 'attract') startAttract();
    const stageButtons = STAGES.map((s, i) => ({
      label: `${i + 1} · ${s.name}`,
      disabled: i >= progress.unlocked,
      action: () => { game.score = 0; startStage(i); },
    }));
    showOverlay({
      title: 'TURBO RALLY',
      body: `<p>Six stages, one clock. Reach each checkpoint before time runs out, and overtake as many of the ${RIVAL_COUNT} rivals as you can.</p>`
        + (progress.best ? `<p class="stats">BEST SCORE ${progress.best}</p>` : '')
        + '<p class="hint"><a href="../turbo-rally-ai/">Watch an AI learn to drive this game →</a> · <a href="../">Back to games</a></p>',
      buttons: [
        [{ label: 'START RACE', primary: true, action: () => { game.score = 0; startStage(0); } }],
        stageButtons,
      ],
    });
  }

  function showStageClear() {
    const r = game.result;
    const last = game.stageIndex === STAGES.length - 1;
    if (last) {
      showOverlay({
        title: 'CHAMPION!',
        body: `<p>You conquered all ${STAGES.length} stages.</p><p class="stats">FINAL SCORE ${game.score}${game.score >= progress.best ? ' · NEW BEST' : ''}</p>`,
        buttons: [[{ label: 'MAIN MENU', primary: true, action: showTitle }]],
      });
      return;
    }
    showOverlay({
      title: 'STAGE CLEAR',
      body: `<p class="stats">POSITION ${r.pos}/${RIVAL_COUNT + 1} · TIME LEFT ${r.timeLeft.toFixed(1)}s · +${r.points} PTS</p>`
        + `<p>Next: Stage ${game.stageIndex + 2}, ${STAGES[game.stageIndex + 1].name}.</p>`,
      buttons: [[
        { label: 'NEXT STAGE', primary: true, action: () => startStage(game.stageIndex + 1) },
        { label: 'MENU', action: showTitle },
      ]],
    });
  }

  function showTimeUp() {
    saveBest();
    showOverlay({
      title: 'TIME UP',
      body: `<p class="stats">STAGE ${game.stageIndex + 1} · SCORE ${game.score}</p><p>Retrying restarts this stage with the score you had when it began.</p>`,
      buttons: [[
        { label: 'RETRY STAGE', primary: true, action: () => { game.score = game.stageStartScore; startStage(game.stageIndex); } },
        { label: 'MENU', action: showTitle },
      ]],
    });
  }

  function togglePause() {
    if (game.state === 'racing' || game.state === 'countdown') {
      game.pausedFrom = game.state;
      game.state = 'paused';
      sound.engine(0, 0, false);
      showOverlay({
        title: 'PAUSED',
        buttons: [[
          { label: 'RESUME', primary: true, action: togglePause },
          { label: 'RESTART STAGE', action: () => { game.score = game.stageStartScore; startStage(game.stageIndex); } },
          { label: 'MENU', action: showTitle },
        ]],
      });
    } else if (game.state === 'paused') {
      game.state = game.pausedFrom;
      hideOverlay();
    }
  }

  // ---------------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------------

  let last = performance.now();
  let acc = 0;

  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (game.state !== 'paused') {
      acc += dt;
      while (acc >= STEP) {
        update(STEP);
        acc -= STEP;
      }
    }
    window.TurboEngine.render(game);
    updateHud();
    requestAnimationFrame(frame);
  }

  // Test hook: ?debug exposes a headless simulator driven by the autopilot.
  if (new URLSearchParams(location.search).has('debug')) {
    window.turboRally = {
      game,
      // Starts a stage with the computer driving, for visual checks.
      watch(stageIndex, seconds = 0) {
        startStage(stageIndex);
        game.autopilot = true;
        game.dryRun = true;
        game.countdown = 0.6;
        for (let t = 0; t < seconds; t += STEP) update(STEP);
      },
      simulate(stageIndex) {
        loadStage(stageIndex);
        game.state = 'racing';
        game.autopilot = true;
        game.dryRun = true;
        const scoreBefore = game.score;
        let elapsed = 0, minTime = Infinity;
        const realCrash = game.crashCooldown;
        while (game.state === 'racing' && elapsed < 900) {
          update(STEP);
          elapsed += STEP;
          minTime = Math.min(minTime, game.timeLeft);
        }
        const result = {
          stage: STAGES[stageIndex].name, outcome: game.state, seconds: Math.round(elapsed),
          timeLeft: Math.round(game.timeLeft * 10) / 10, closestCall: Math.round(minTime * 10) / 10,
          position: position(), hits: { ...game.hits }, budgets: game.track.budgets, segments: game.track.finish,
        };
        game.crashCooldown = realCrash;
        game.score = scoreBefore;
        game.dryRun = false;
        startAttract();
        showTitle();
        return result;
      },
    };
  }

  // ---------------------------------------------------------------------------
  // Screen size, fullscreen and orientation
  // ---------------------------------------------------------------------------

  const screenEl = $('screen');

  function resizeView() {
    const rect = screenEl.getBoundingClientRect();
    if (window.TurboEngine.resizeView(rect.width, rect.height) && game.theme) resetWeather(game.theme);
  }
  new ResizeObserver(resizeView).observe(screenEl);

  // Phones are landscape-only: pause a race if the phone is turned upright.
  const portrait = matchMedia('(pointer: coarse) and (orientation: portrait)');
  portrait.addEventListener('change', () => {
    if (portrait.matches && (game.state === 'racing' || game.state === 'countdown')) togglePause();
  });

  const fsButton = $('fullscreen');
  const fsSupported = !!(document.fullscreenEnabled || document.webkitFullscreenEnabled);
  const fsElement = () => document.fullscreenElement || document.webkitFullscreenElement;
  fsButton.hidden = !fsSupported; // iPhone Safari only allows fullscreen video

  function toggleFullscreen() {
    if (!fsSupported) return;
    if (fsElement()) {
      (document.exitFullscreen || document.webkitExitFullscreen).call(document);
      return;
    }
    const request = screenEl.requestFullscreen || screenEl.webkitRequestFullscreen;
    const result = request.call(screenEl, { navigationUI: 'hide' });
    // Orientation lock only works while fullscreen, and only on some phones.
    const lock = () => screen.orientation?.lock?.('landscape').catch(() => {});
    if (result && result.then) result.then(lock, () => {});
    else lock();
  }

  fsButton.addEventListener('click', (e) => {
    e.stopPropagation();
    sound.init();
    toggleFullscreen();
  });
  for (const ev of ['fullscreenchange', 'webkitfullscreenchange']) {
    document.addEventListener(ev, () => fsButton.classList.toggle('active', !!fsElement()));
  }

  resizeView();
  startAttract();
  showTitle();
  requestAnimationFrame((t) => { last = t; frame(t); });
})();
