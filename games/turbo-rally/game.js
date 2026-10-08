(() => {
  'use strict';

  // ===========================================================================
  // Turbo Rally: a pseudo-3D stage racer in the spirit of the 16-bit classics.
  // The road technique (projected road segments, curves as accumulated x
  // offsets, hills as segment heights, sprites scaled per segment) follows
  // Louis Gorenfeld's "Lou's Pseudo 3d Page" and Jake Gordon's write-ups.
  // ===========================================================================

  // ---------------------------------------------------------------------------
  // Constants
  // ---------------------------------------------------------------------------

  const W = 480, H = 270;              // internal resolution, scaled up by CSS
  const SEG = 200;                     // segment length (world units)
  const RUMBLE_LENGTH = 3;             // segments per rumble strip colour band
  const ROAD_WIDTH = 2000;             // half the road width (world units)
  const LANES = 3;
  const CAMERA_HEIGHT = 1000;
  const CAMERA_DEPTH = 1 / Math.tan((100 / 2) * Math.PI / 180); // 100° FOV
  const DRAW_DISTANCE = 300;           // segments drawn ahead
  const PLAYER_Z = CAMERA_HEIGHT * CAMERA_DEPTH;
  const STEP = 1 / 60;
  const MAX_SPEED = SEG / STEP;        // one segment per physics step
  const ACCEL = MAX_SPEED / 5;
  const BRAKING = -MAX_SPEED;
  const DECEL = -MAX_SPEED / 5;
  const OFFROAD_DECEL = -MAX_SPEED / 2;
  const OFFROAD_LIMIT = MAX_SPEED / 4;
  const CENTRIFUGAL = 0.3;
  const TOP_KMH = 240;
  const GEARS = 5;
  const RIVAL_COUNT = 19;
  const CAR_WORLD_W = 560;
  const CAR_W = CAR_WORLD_W / ROAD_WIDTH; // car width in road half-widths

  const ROAD = { SHORT: 25, MEDIUM: 50, LONG: 100 };
  const CURVE = { EASY: 2, MEDIUM: 4, HARD: 6 };
  const HILL = { LOW: 20, MEDIUM: 40, HIGH: 60 };

  // ---------------------------------------------------------------------------
  // Helpers
  // ---------------------------------------------------------------------------

  function mulberry32(a) {
    return () => {
      a = (a + 0x6d2b79f5) | 0;
      let t = Math.imul(a ^ (a >>> 15), 1 | a);
      t = (t + Math.imul(t ^ (t >>> 7), 61 | t)) ^ t;
      return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
    };
  }

  const clamp = (v, lo, hi) => Math.max(lo, Math.min(hi, v));
  const lerp = (a, b, p) => a + (b - a) * p;
  const easeIn = (a, b, p) => a + (b - a) * p * p;
  const easeInOut = (a, b, p) => a + (b - a) * (-Math.cos(p * Math.PI) / 2 + 0.5);
  const fogAt = (distance, density) => 1 / Math.exp(distance * distance * density);

  function overlap(x1, w1, x2, w2, percent = 1) {
    const half = percent / 2;
    return !(x1 + w1 * half < x2 - w2 * half || x1 - w1 * half > x2 + w2 * half);
  }

  function weightedPick(rng, entries) {
    const total = entries.reduce((s, [, w]) => s + w, 0);
    let r = rng() * total;
    for (const [value, w] of entries) {
      if ((r -= w) < 0) return value;
    }
    return entries[entries.length - 1][0];
  }

  // ---------------------------------------------------------------------------
  // Sprites, all drawn procedurally into small offscreen canvases
  // ---------------------------------------------------------------------------

  function makeCanvas(w, h, draw) {
    const c = document.createElement('canvas');
    c.width = w;
    c.height = h;
    draw(c.getContext('2d'), w, h);
    return c;
  }

  function poly(g, color, ...pts) {
    g.fillStyle = color;
    g.beginPath();
    g.moveTo(pts[0], pts[1]);
    for (let i = 2; i < pts.length; i += 2) g.lineTo(pts[i], pts[i + 1]);
    g.closePath();
    g.fill();
  }

  function ellipse(g, color, x, y, rx, ry) {
    g.fillStyle = color;
    g.beginPath();
    g.ellipse(x, y, rx, ry, 0, 0, Math.PI * 2);
    g.fill();
  }

  function shade(hex, amt) {
    const n = parseInt(hex.slice(1), 16);
    const c = (v) => clamp(v + amt, 0, 255);
    return `rgb(${c(n >> 16)},${c((n >> 8) & 255)},${c(n & 255)})`;
  }

  const draw = {
    pine(g, w, h, snowy) {
      g.fillStyle = '#5a3a1e';
      g.fillRect(w / 2 - 3, h - 14, 6, 14);
      const greens = ['#165c33', '#1d7040', '#24854d'];
      for (let i = 0; i < 3; i++) {
        const top = h * (0.5 - i * 0.22);
        const bottom = top + h * 0.42;
        const half = w * (0.48 - i * 0.12);
        poly(g, greens[i], w / 2, top, w / 2 + half, bottom, w / 2 - half, bottom);
        if (snowy) {
          const sy = top + (bottom - top) * 0.45;
          poly(g, '#f4f8ff', w / 2, top, w / 2 + half * 0.45, sy, w / 2 - half * 0.45, sy);
        }
      }
    },
    tree(g, w, h) {
      g.fillStyle = '#6b4423';
      g.fillRect(w / 2 - 4, h * 0.55, 8, h * 0.45);
      ellipse(g, '#2a7a35', w / 2, h * 0.38, w * 0.46, h * 0.3);
      ellipse(g, '#33913f', w * 0.38, h * 0.3, w * 0.26, h * 0.2);
      ellipse(g, '#3aa548', w * 0.6, h * 0.24, w * 0.22, h * 0.17);
    },
    bush(g, w, h) {
      ellipse(g, '#2a7a35', w * 0.3, h * 0.62, w * 0.3, h * 0.38);
      ellipse(g, '#33913f', w * 0.65, h * 0.6, w * 0.33, h * 0.4);
      ellipse(g, '#3aa548', w * 0.48, h * 0.42, w * 0.26, h * 0.3);
    },
    cactus(g, w, h) {
      g.fillStyle = '#2f8a3a';
      const r = (x, y, ww, hh) => { g.beginPath(); g.roundRect(x, y, ww, hh, ww / 2); g.fill(); };
      r(w / 2 - 5, 2, 10, h - 2);
      r(4, h * 0.35, 8, h * 0.3);
      r(4, h * 0.55, w / 2 - 6, 7);
      r(w - 12, h * 0.22, 8, h * 0.3);
      r(w / 2, h * 0.45, w / 2 - 6, 7);
      g.fillStyle = '#3fa64b';
      g.fillRect(w / 2 - 2, 6, 2, h - 10);
    },
    rock(g, w, h, snowy) {
      poly(g, '#7d7468', 2, h, 6, h * 0.45, w * 0.35, h * 0.15, w * 0.7, h * 0.25, w - 3, h * 0.6, w - 1, h);
      poly(g, '#958b7e', w * 0.35, h * 0.15, w * 0.7, h * 0.25, w * 0.55, h * 0.55, w * 0.25, h * 0.5);
      if (snowy) poly(g, '#f4f8ff', 6, h * 0.45, w * 0.35, h * 0.15, w * 0.7, h * 0.25, w - 3, h * 0.6, w * 0.5, h * 0.4, w * 0.2, h * 0.55);
    },
    deadTree(g, w, h) {
      g.strokeStyle = '#6b4a2e';
      g.lineCap = 'round';
      const branch = (x, y, len, angle, width) => {
        if (width < 1) return;
        const x2 = x + Math.cos(angle) * len, y2 = y + Math.sin(angle) * len;
        g.lineWidth = width;
        g.beginPath(); g.moveTo(x, y); g.lineTo(x2, y2); g.stroke();
        branch(x2, y2, len * 0.65, angle - 0.5, width * 0.6);
        branch(x2, y2, len * 0.6, angle + 0.45, width * 0.6);
      };
      branch(w / 2, h, h * 0.42, -Math.PI / 2, 6);
    },
    palm(g, w, h) {
      g.strokeStyle = '#8a6a3e';
      g.lineWidth = 5;
      g.beginPath();
      g.moveTo(w * 0.45, h);
      g.quadraticCurveTo(w * 0.42, h * 0.5, w * 0.55, h * 0.22);
      g.stroke();
      g.strokeStyle = '#2f8a3a';
      g.lineWidth = 4;
      for (const a of [-2.8, -2.2, -1.6, -1.0, -0.35]) {
        g.beginPath();
        g.moveTo(w * 0.55, h * 0.22);
        g.quadraticCurveTo(w * 0.55 + Math.cos(a) * w * 0.3, h * 0.22 + Math.sin(a) * h * 0.12,
          w * 0.55 + Math.cos(a) * w * 0.45, h * 0.3 + Math.sin(a) * h * 0.02);
        g.stroke();
      }
    },
    lamp(g, w, h) {
      g.fillStyle = '#4a4f5a';
      g.fillRect(w / 2 - 2, 10, 4, h - 10);
      const glow = g.createRadialGradient(w / 2, 9, 1, w / 2, 9, w / 2);
      glow.addColorStop(0, 'rgba(255,240,180,1)');
      glow.addColorStop(1, 'rgba(255,220,120,0)');
      g.fillStyle = glow;
      g.fillRect(0, 0, w, 20);
      ellipse(g, '#fff6c8', w / 2, 9, 4, 3);
    },
    chevron(g, w, h, dir) {
      g.fillStyle = '#4a4f5a';
      g.fillRect(w / 2 - 2, h * 0.5, 4, h * 0.5);
      g.fillStyle = '#ffd400';
      g.fillRect(1, 2, w - 2, h * 0.55);
      g.fillStyle = '#111';
      const cy = 2 + h * 0.275, s = h * 0.18;
      for (const ox of [-w * 0.2, w * 0.12]) {
        const x = w / 2 + ox * dir;
        poly(g, '#111', x - s * dir, cy - s, x, cy - s, x + s * dir, cy, x, cy + s, x - s * dir, cy + s, x, cy);
      }
    },
    billboard(g, w, h, text, bg, fg) {
      g.fillStyle = '#4a4f5a';
      g.fillRect(w * 0.18, h * 0.55, 5, h * 0.45);
      g.fillRect(w * 0.78, h * 0.55, 5, h * 0.45);
      g.fillStyle = '#222';
      g.fillRect(0, 0, w, h * 0.62);
      g.fillStyle = bg;
      g.fillRect(3, 3, w - 6, h * 0.62 - 6);
      g.fillStyle = fg;
      g.font = `bold italic ${Math.floor(h * 0.3)}px sans-serif`;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, w / 2, h * 0.31);
    },
    snowman(g, w, h) {
      ellipse(g, '#f4f8ff', w / 2, h * 0.75, w * 0.42, h * 0.24);
      ellipse(g, '#f4f8ff', w / 2, h * 0.38, w * 0.3, h * 0.18);
      g.fillStyle = '#222';
      g.fillRect(w * 0.3, h * 0.08, w * 0.4, h * 0.14);
      g.fillRect(w * 0.2, h * 0.2, w * 0.6, h * 0.03);
      poly(g, '#ff8a1a', w / 2, h * 0.38, w * 0.75, h * 0.41, w / 2, h * 0.43);
    },
    gantry(g, w, h, text, checkered) {
      g.fillStyle = '#d0d4dc';
      g.fillRect(w * 0.06, h * 0.2, 6, h * 0.8);
      g.fillRect(w * 0.94 - 6, h * 0.2, 6, h * 0.8);
      const top = h * 0.12, bh = h * 0.24;
      if (checkered) {
        const s = bh / 3;
        for (let x = w * 0.06; x < w * 0.94; x += s) {
          for (let r = 0; r < 3; r++) {
            g.fillStyle = (Math.floor(x / s) + r) % 2 ? '#111' : '#fff';
            g.fillRect(x, top + r * s, s + 0.5, s + 0.5);
          }
        }
      } else {
        g.fillStyle = '#d83a1e';
        g.fillRect(w * 0.06, top, w * 0.88, bh);
      }
      g.fillStyle = checkered ? 'rgba(0,0,0,0.75)' : 'rgba(0,0,0,0)';
      g.fillRect(w * 0.3, top + 2, w * 0.4, bh - 4);
      g.fillStyle = '#fff';
      g.font = `bold ${Math.floor(bh * 0.65)}px sans-serif`;
      g.textAlign = 'center';
      g.textBaseline = 'middle';
      g.fillText(text, w / 2, top + bh / 2 + 1);
    },
    car(g, w, h, color, lean) {
      const L = lean * 3;
      ellipse(g, 'rgba(0,0,0,0.35)', w / 2, h - 3, w * 0.48, 4);
      g.fillStyle = '#111';
      g.fillRect(3, h - 15, 15, 13);
      g.fillRect(w - 18, h - 15, 15, 13);
      poly(g, color, 1, h - 8, w - 1, h - 8, w - 3, h - 25, 3, h - 25);
      g.fillStyle = shade(color, -45);
      g.fillRect(4, h - 12, w - 8, 4);
      g.fillStyle = '#ff2a2a';
      g.fillRect(6, h - 21, 15, 5);
      g.fillRect(w - 21, h - 21, 15, 5);
      g.fillStyle = '#ffb0a0';
      g.fillRect(8, h - 20, 4, 2);
      g.fillRect(w - 19, h - 20, 4, 2);
      g.fillStyle = '#e8e8e8';
      g.fillRect(w / 2 - 8, h - 19, 16, 6);
      poly(g, shade(color, 25), 14 + L, h - 25, w - 14 + L, h - 25, w - 22 + L * 1.6, h - 37, 22 + L * 1.6, h - 37);
      poly(g, '#1a2333', 18 + L, h - 26, w - 18 + L, h - 26, w - 24 + L * 1.6, h - 35, 24 + L * 1.6, h - 35);
      g.fillStyle = 'rgba(255,255,255,0.25)';
      g.fillRect(26 + L * 1.6, h - 34, 10, 2);
      g.fillStyle = shade(color, -60);
      g.fillRect(5 + L, h - 30, w - 10, 3);
      g.fillRect(12 + L, h - 28, 3, 4);
      g.fillRect(w - 15 + L, h - 28, 3, 4);
    },
  };

  // worldW: width in world units; hit: fraction of the width that collides.
  function sprite(w, h, worldW, hit, fn, ...args) {
    return { canvas: makeCanvas(w, h, (g) => fn(g, w, h, ...args)), worldW, hit };
  }

  const PLAYER_COLOR = '#d81e1e';
  const RIVAL_COLORS = ['#1e5bd8', '#f0c419', '#f2f2f2', '#1fa04a', '#2a2a2a', '#ff7a1a', '#7a2bd8', '#18b5c4'];

  const SPRITES = {
    pine: sprite(48, 96, 1300, 0.3, draw.pine, false),
    snowPine: sprite(48, 96, 1300, 0.3, draw.pine, true),
    tree: sprite(64, 80, 1500, 0.3, draw.tree),
    bush: sprite(48, 28, 800, 0.8, draw.bush),
    cactus: sprite(32, 64, 600, 0.5, draw.cactus),
    rock: sprite(48, 32, 800, 0.8, draw.rock, false),
    snowRock: sprite(48, 32, 800, 0.8, draw.rock, true),
    deadTree: sprite(48, 64, 1000, 0.25, draw.deadTree),
    palm: sprite(64, 96, 1400, 0.25, draw.palm),
    lamp: sprite(24, 112, 380, 0.4, draw.lamp),
    snowman: sprite(32, 48, 500, 0.7, draw.snowman),
    chevronL: sprite(32, 40, 420, 0.6, draw.chevron, -1),
    chevronR: sprite(32, 40, 420, 0.6, draw.chevron, 1),
    ad1: sprite(96, 64, 1700, 0.9, draw.billboard, 'TURBO', '#ffd84a', '#d81e1e'),
    ad2: sprite(96, 64, 1700, 0.9, draw.billboard, 'PIXEL TYRES', '#1e5bd8', '#fff'),
    ad3: sprite(96, 64, 1700, 0.9, draw.billboard, 'NITRO OIL', '#1fa04a', '#fff'),
    start: sprite(256, 96, ROAD_WIDTH * 2.5, 0, draw.gantry, 'START', true),
    checkpoint: sprite(256, 96, ROAD_WIDTH * 2.5, 0, draw.gantry, 'CHECKPOINT', false),
    finish: sprite(256, 96, ROAD_WIDTH * 2.5, 0, draw.gantry, 'FINISH', true),
  };

  const carSprites = (color) => ({
    straight: sprite(80, 44, CAR_WORLD_W, 1, draw.car, color, 0),
    left: sprite(80, 44, CAR_WORLD_W, 1, draw.car, color, -1),
    right: sprite(80, 44, CAR_WORLD_W, 1, draw.car, color, 1),
  });
  const PLAYER_SPRITES = carSprites(PLAYER_COLOR);
  const RIVAL_SPRITES = RIVAL_COLORS.map((c) => carSprites(c).straight);

  // ---------------------------------------------------------------------------
  // Themes & stages
  // ---------------------------------------------------------------------------

  const THEMES = {
    forest: {
      sky: ['#3d7fd6', '#a9d4f5'], far: '#7d96b8', near: '#2f6b3a', treeline: '#24572f',
      ground: ['#3f9b3f', '#378c37'], rumble: ['#e8e8e8', '#c0392b'], road: ['#6b6b6b', '#656565'],
      lane: '#e8e8e8', fog: '#a9d4f5', fogDensity: 4, grip: 1,
      scenery: [['pine', 5], ['tree', 3], ['bush', 2]], density: 0.75,
    },
    desert: {
      sky: ['#f08a3a', '#fde3a0'], far: '#c47f4a', near: '#d9a066',
      ground: ['#e2c07d', '#d9b46e'], rumble: ['#f2f2f2', '#d35400'], road: ['#7a6a5a', '#736353'],
      lane: '#f2f2f2', fog: '#fde3a0', fogDensity: 3, grip: 1,
      scenery: [['cactus', 4], ['rock', 3], ['deadTree', 2], ['palm', 1]], density: 0.35,
    },
    night: {
      sky: ['#03040c', '#18203f'], stars: true, far: '#121630', near: '#0b0e1d', treeline: '#070912',
      ground: ['#0e2a17', '#0c2514'], rumble: ['#b8b8b8', '#8a1c1c'], road: ['#2d2d35', '#29292f'],
      lane: '#d8d8d8', fog: '#03040c', fogDensity: 8, grip: 1, night: true, lamps: true,
      scenery: [['pine', 4], ['tree', 2]], density: 0.5,
    },
    fog: {
      sky: ['#9aa3ab', '#c4c9cd'], far: '#b2b8bd', near: '#a3aaaf',
      ground: ['#5e7d5a', '#58764f'], rumble: ['#dddddd', '#999999'], road: ['#6f7174', '#6a6c6f'],
      lane: '#dddddd', fog: '#c4c9cd', fogDensity: 26, grip: 0.95,
      scenery: [['tree', 3], ['pine', 3], ['bush', 2]], density: 0.6,
    },
    snow: {
      sky: ['#8fb2d6', '#dfe9f3'], far: '#eef3f8', snowcaps: true, near: '#c9d6e3', treeline: '#5c7a6a',
      ground: ['#f4f7fb', '#e7edf4'], rumble: ['#ffffff', '#2c5aa0'], road: ['#8c9298', '#868c92'],
      lane: '#ffffff', fog: '#dfe9f3', fogDensity: 7, grip: 0.7, weather: 'snow',
      scenery: [['snowPine', 5], ['snowRock', 2], ['snowman', 0.3]], density: 0.6,
    },
    mountain: {
      sky: ['#4b5868', '#8a97a6'], far: '#5f6877', snowcaps: true, near: '#4d6b4a', treeline: '#3a5538',
      ground: ['#5b8a4c', '#548046'], rumble: ['#eeeeee', '#c0392b'], road: ['#5d5f63', '#595b5f'],
      lane: '#e0e0e0', fog: '#8a97a6', fogDensity: 6, grip: 0.85, weather: 'rain',
      scenery: [['rock', 3], ['pine', 3], ['bush', 1]], density: 0.5,
    },
  };

  // length: segments between checkpoints; pace: fraction of top speed you must
  // average to reach each checkpoint in time; rivals: their speed range.
  const STAGES = [
    { name: 'Forest', theme: 'forest', seed: 101, curviness: 0.55, hilliness: 0.5, length: 1300, checkpoints: 3, pace: 0.68, rivals: [0.6, 0.9] },
    { name: 'Desert', theme: 'desert', seed: 202, curviness: 0.6, hilliness: 0.35, length: 1400, checkpoints: 3, pace: 0.7, rivals: [0.62, 0.91] },
    { name: 'Night', theme: 'night', seed: 303, curviness: 0.7, hilliness: 0.5, length: 1400, checkpoints: 3, pace: 0.68, rivals: [0.62, 0.91] },
    { name: 'Fog', theme: 'fog', seed: 404, curviness: 0.7, hilliness: 0.6, length: 1400, checkpoints: 3, pace: 0.7, rivals: [0.63, 0.92] },
    { name: 'Snow', theme: 'snow', seed: 505, curviness: 0.65, hilliness: 0.6, length: 1400, checkpoints: 3, pace: 0.64, rivals: [0.62, 0.9] },
    { name: 'Mountains', theme: 'mountain', seed: 606, curviness: 0.85, hilliness: 1, length: 1500, checkpoints: 4, pace: 0.7, rivals: [0.65, 0.94] },
  ];

  // ---------------------------------------------------------------------------
  // Track building
  // ---------------------------------------------------------------------------

  class Track {
    constructor() {
      this.segments = [];
      this.curves = [];
    }

    lastY() {
      const n = this.segments.length;
      return n ? this.segments[n - 1].p2.world.y : 0;
    }

    addSegment(curve, y) {
      const n = this.segments.length;
      this.segments.push({
        index: n,
        p1: { world: { y: this.lastY(), z: n * SEG }, camera: {}, screen: {} },
        p2: { world: { y, z: (n + 1) * SEG }, camera: {}, screen: {} },
        curve,
        sprites: [],
        cars: [],
        dark: Math.floor(n / RUMBLE_LENGTH) % 2 === 1,
        clip: 0,
        fog: 1,
        line: null,
      });
    }

    addRoad(enter, hold, leave, curve, hill) {
      const startY = this.lastY();
      const endY = startY + hill * SEG;
      const total = enter + hold + leave;
      if (Math.abs(curve) >= 2.5) {
        this.curves.push({ start: this.segments.length, length: total, dir: Math.sign(curve) });
      }
      for (let n = 0; n < enter; n++) this.addSegment(easeIn(0, curve, n / enter), easeInOut(startY, endY, n / total));
      for (let n = 0; n < hold; n++) this.addSegment(curve, easeInOut(startY, endY, (enter + n) / total));
      for (let n = 0; n < leave; n++) this.addSegment(easeInOut(curve, 0, n / leave), easeInOut(startY, endY, (enter + hold + n) / total));
    }

    straight(n) { this.addRoad(0, n, 0, 0, 0); }

    find(z) {
      return this.segments[clamp(Math.floor(z / SEG), 0, this.segments.length - 1)];
    }
  }

  function addRandomSection(t, rng, stage) {
    const dir = rng() < 0.5 ? -1 : 1;
    const curve = dir * (CURVE.EASY + rng() * (CURVE.HARD - CURVE.EASY)) * stage.curviness;
    const hill = (rng() * 2 - 1) * HILL.HIGH * stage.hilliness;
    const len = [ROAD.SHORT, ROAD.MEDIUM, ROAD.MEDIUM, ROAD.LONG][Math.floor(rng() * 4)];
    const r = rng();
    if (r < 0.18) {
      t.addRoad(len, len, len, 0, hill * 0.5);
    } else if (r < 0.55) {
      t.addRoad(len, len, len, curve, hill * 0.6);
    } else if (r < 0.7) {
      t.addRoad(ROAD.MEDIUM, ROAD.SHORT, ROAD.MEDIUM, curve, 0);
      t.addRoad(ROAD.MEDIUM, ROAD.SHORT, ROAD.MEDIUM, -curve, 0);
    } else if (r < 0.85) {
      t.addRoad(len, len, len, 0, hill);
    } else {
      // A run of small crests.
      for (let i = 0; i < 4; i++) t.addRoad(10, 10, 10, curve * 0.3, (i % 2 ? -1 : 1) * HILL.LOW * 0.25 * stage.hilliness * 2);
    }
  }

  function buildTrack(stage) {
    const rng = mulberry32(stage.seed);
    const theme = THEMES[stage.theme];
    const t = new Track();

    t.straight(40);
    t.segments[6].sprites.push({ name: 'start', offset: 0, center: true });
    t.segments[6].line = 'checkered';

    t.checkpoints = [];
    for (let cp = 0; cp < stage.checkpoints; cp++) {
      const target = t.segments.length + stage.length;
      while (t.segments.length < target) addRandomSection(t, rng, stage);
      t.straight(30);
      const idx = t.segments.length - 15;
      const last = cp === stage.checkpoints - 1;
      t.segments[idx].sprites.push({ name: last ? 'finish' : 'checkpoint', offset: 0, center: true });
      t.segments[idx].line = last ? 'checkered' : 'white';
      t.checkpoints.push(idx);
    }
    t.finish = t.checkpoints[t.checkpoints.length - 1];
    t.straight(DRAW_DISTANCE + 200); // run-off so the renderer never runs out of road

    // Seconds granted for each section, from its length and the stage's pace.
    t.budgets = t.checkpoints.map((idx, i) => {
      const from = i === 0 ? 0 : t.checkpoints[i - 1];
      return Math.round((idx - from) / (60 * stage.pace)) + 2;
    });

    // Chevron signs on the outside of the sharper bends.
    for (const c of t.curves) {
      for (let i = c.start; i < c.start + c.length; i += 8) {
        t.segments[i].sprites.push({ name: c.dir > 0 ? 'chevronR' : 'chevronL', offset: -c.dir * 1.2, collide: true });
      }
    }

    // Scenery along both sides.
    const end = t.segments.length - 1;
    for (let n = 14; n < end; n += 2 + Math.floor(rng() * 3)) {
      for (const side of [-1, 1]) {
        if (rng() > theme.density) continue;
        const name = weightedPick(rng, theme.scenery);
        t.segments[n].sprites.push({ name, offset: side * (1.35 + rng() * rng() * 3.5), collide: true });
      }
    }
    for (let n = 60; n < end; n += 140 + Math.floor(rng() * 120)) {
      const side = rng() < 0.5 ? -1 : 1;
      t.segments[n].sprites.push({ name: `ad${1 + Math.floor(rng() * 3)}`, offset: side * 1.5, collide: true });
    }
    if (theme.lamps) {
      for (let n = 20; n < end; n += 24) {
        t.segments[n].sprites.push({ name: 'lamp', offset: (n % 48 ? 1 : -1) * 1.15, collide: true });
      }
    }
    return t;
  }

  // ---------------------------------------------------------------------------
  // Backgrounds: sky plus two parallax silhouette layers per theme
  // ---------------------------------------------------------------------------

  const backgrounds = {};

  function silhouetteLayer(seed, color, base, amp, opts = {}) {
    const lw = W * 2;
    return makeCanvas(lw, H, (g) => {
      const rng = mulberry32(seed);
      // Whole-number frequencies make the layer tile seamlessly.
      const waves = [1, 2, 3, 5, 8, 13].map((f) => ({ f, a: rng() / Math.sqrt(f), ph: rng() * Math.PI * 2 }));
      const norm = waves.reduce((s, w) => s + w.a, 0);
      const yAt = (x) => base - amp * waves.reduce((s, w) => s + w.a * (Math.sin((2 * Math.PI * w.f * x) / lw + w.ph) + 1) / 2, 0) / norm;
      const outline = () => {
        g.beginPath();
        g.moveTo(0, H);
        for (let x = 0; x <= lw; x += 3) g.lineTo(x, yAt(x));
        g.lineTo(lw, H);
        g.closePath();
      };
      g.fillStyle = color;
      outline();
      g.fill();
      if (opts.snowcaps) {
        g.save();
        outline();
        g.clip();
        g.fillStyle = '#f4f8ff';
        g.fillRect(0, 0, lw, base - amp * 0.62);
        g.restore();
      }
      if (opts.treeline) {
        g.fillStyle = opts.treeline;
        for (let x = 0; x < lw; x += 5) {
          const y = yAt(x) + 2, s = 4 + rng() * 6;
          poly(g, opts.treeline, x, y - s * 1.8, x + s / 2, y, x - s / 2, y);
        }
      }
    });
  }

  function getBackground(themeName) {
    if (backgrounds[themeName]) return backgrounds[themeName];
    const th = THEMES[themeName];
    const sky = makeCanvas(W, H, (g) => {
      const grad = g.createLinearGradient(0, 0, 0, H * 0.62);
      grad.addColorStop(0, th.sky[0]);
      grad.addColorStop(1, th.sky[1]);
      g.fillStyle = grad;
      g.fillRect(0, 0, W, H);
      if (th.stars) {
        const rng = mulberry32(7);
        for (let i = 0; i < 120; i++) {
          g.fillStyle = `rgba(255,255,255,${0.3 + rng() * 0.7})`;
          g.fillRect(Math.floor(rng() * W), Math.floor(rng() * H * 0.55), 1, 1);
        }
        ellipse(g, '#f2f0d8', W * 0.78, H * 0.16, 11, 11);
        ellipse(g, th.sky[0], W * 0.78 + 5, H * 0.16 - 3, 9, 9);
      }
    });
    backgrounds[themeName] = {
      sky,
      far: silhouetteLayer(themeName.length * 17 + 3, th.far, H * 0.56, H * 0.3, { snowcaps: th.snowcaps }),
      near: silhouetteLayer(themeName.length * 31 + 9, th.near, H * 0.62, H * 0.14, { treeline: th.treeline }),
    };
    return backgrounds[themeName];
  }

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
  const ctx = canvas.getContext('2d');
  ctx.imageSmoothingEnabled = false;
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

  const particles = [];

  function loadStage(index, { attract = false } = {}) {
    const stage = STAGES[index];
    game.stageIndex = index;
    game.stage = stage;
    game.theme = THEMES[stage.theme];
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
    resetWeather();
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

  function resetWeather() {
    particles.length = 0;
    const kind = game.theme.weather;
    if (!kind) return;
    const rng = mulberry32(42);
    for (let i = 0; i < (kind === 'snow' ? 140 : 170); i++) {
      particles.push({ x: rng() * W, y: rng() * H, z: 0.3 + rng() * 0.7 });
    }
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
    p: 'pause', escape: 'pause', m: 'mute', enter: 'enter',
  };
  const CODES = {
    ArrowLeft: 'left', KeyA: 'left', ArrowRight: 'right', KeyD: 'right',
    ArrowUp: 'up', KeyW: 'up', ArrowDown: 'down', KeyS: 'down',
    KeyP: 'pause', Escape: 'pause', KeyM: 'mute', Enter: 'enter',
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

  // The computer driver, used for the title-screen demo and for testing.
  function autopilotInput(playerSeg) {
    const segs = game.track.segments;
    const speedPct = game.speed / MAX_SPEED;
    let target = 0;
    search:
    for (let i = 1; i < 25; i++) {
      const seg = segs[Math.min(segs.length - 1, playerSeg.index + i)];
      for (const car of seg.cars) {
        if (car.speed < game.speed && overlap(game.playerX, CAR_W, car.offset, CAR_W, 1.6)) {
          target = car.offset > 0 ? car.offset - 0.7 : car.offset + 0.7;
          break search;
        }
      }
    }
    target = clamp(target, -0.75, 0.75);
    let maxCurve = 0;
    for (let i = 0; i < 30; i++) {
      maxCurve = Math.max(maxCurve, Math.abs(segs[Math.min(segs.length - 1, playerSeg.index + i)].curve));
    }
    const drift = maxCurve * CENTRIFUGAL * speedPct * (2 - game.theme.grip);
    return {
      left: game.playerX > target + 0.04,
      right: game.playerX < target - 0.04,
      up: drift < 0.85,
      down: drift > 1.25,
    };
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
    if (game.autopilot || game.state === 'finished') controls = autopilotInput(playerSeg);
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
      updateWeather(dt, playerSeg, 0);
      return;
    }

    updateRivals(dt, playerSeg);

    game.position = Math.min(game.position + dt * game.speed, (t.segments.length - DRAW_DISTANCE - 10) * SEG);

    game.steer = controls.left ? -1 : controls.right ? 1 : 0;
    game.playerX += game.steer * dx * (0.85 + 0.15 * th.grip);
    game.playerX -= dx * speedPct * playerSeg.curve * CENTRIFUGAL * (2 - th.grip);

    game.throttle = controls.up ? 1 : 0;
    if (controls.up) game.speed += ACCEL * dt;
    else if (controls.down) game.speed += BRAKING * dt;
    else game.speed += DECEL * dt;

    if (Math.abs(game.playerX) > 1) {
      if (game.speed > OFFROAD_LIMIT) game.speed += OFFROAD_DECEL * dt;
      for (const s of playerSeg.sprites) {
        if (!s.collide) continue;
        const spr = SPRITES[s.name];
        const sw = spr.worldW / ROAD_WIDTH;
        const center = s.offset + (sw / 2) * (s.offset > 0 ? 1 : -1);
        if (overlap(game.playerX, CAR_W, center, sw * spr.hit)) {
          // Resting against an obstacle shouldn't keep re-triggering a crash.
          const hard = game.speed > MAX_SPEED / 5;
          game.speed = Math.min(game.speed, MAX_SPEED / 5);
          game.position = Math.max(0, playerSeg.p1.world.z - PLAYER_Z);
          if (hard) {
            game.hits.scenery++;
            crash();
          }
          break;
        }
      }
    }

    for (const car of playerSeg.cars) {
      if (game.speed > car.speed && overlap(game.playerX, CAR_W, car.offset, CAR_W, 0.8)) {
        game.speed = car.speed * (car.speed / game.speed);
        game.position = car.z - PLAYER_Z;
        game.hits.cars++;
        crash(0.15);
        break;
      }
    }

    game.playerX = clamp(game.playerX, -2.5, 2.5);
    game.speed = clamp(game.speed, 0, MAX_SPEED);
    game.crashCooldown -= dt;
    game.shake = Math.max(0, game.shake - dt);

    const travelled = (game.position - startPosition) / SEG;
    game.skyOffset = (game.skyOffset + 0.0012 * playerSeg.curve * travelled + 1) % 1;
    game.nearOffset = (game.nearOffset + 0.0024 * playerSeg.curve * travelled + 1) % 1;

    updateWeather(dt, playerSeg, speedPct);
    updateRace(dt);

    const { rpm } = gearbox();
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

  function updateWeather(dt, playerSeg, speedPct) {
    const kind = game.theme.weather;
    if (!kind) return;
    const sway = -playerSeg.curve * speedPct * 30 - game.steer * speedPct * 25;
    for (const p of particles) {
      if (kind === 'snow') {
        p.y += (16 + p.z * 34) * (1 + speedPct) * dt;
        p.x += (sway + Math.sin(p.y * 0.05 + p.z * 10) * 8) * p.z * dt;
      } else {
        p.y += (220 + p.z * 220) * dt;
        p.x += (sway - 50) * dt;
      }
      if (p.y > H) { p.y -= H; p.x = Math.random() * W; }
      if (p.x < 0) p.x += W;
      if (p.x > W) p.x -= W;
    }
  }

  function gearbox() {
    const pct = game.speed / MAX_SPEED;
    const raw = pct * GEARS;
    const gear = clamp(Math.floor(raw) + 1, 1, GEARS);
    const within = clamp(raw - (gear - 1), 0, 1);
    return { gear, rpm: clamp(0.25 + within * 0.75 + (pct >= 0.99 ? 0.05 : 0), 0, 1) };
  }

  // ---------------------------------------------------------------------------
  // Rendering
  // ---------------------------------------------------------------------------

  function project(p, camX, camY, camZ) {
    p.camera.x = -camX;
    p.camera.y = p.world.y - camY;
    p.camera.z = p.world.z - camZ;
    p.screen.scale = CAMERA_DEPTH / p.camera.z;
    p.screen.x = Math.round(W / 2 + (p.screen.scale * p.camera.x * W) / 2);
    p.screen.y = Math.round(H / 2 - (p.screen.scale * p.camera.y * H) / 2);
    p.screen.w = Math.round((p.screen.scale * ROAD_WIDTH * W) / 2);
  }

  function quad(color, x1, y1, x2, y2, x3, y3, x4, y4) {
    ctx.fillStyle = color;
    ctx.beginPath();
    ctx.moveTo(x1, y1);
    ctx.lineTo(x2, y2);
    ctx.lineTo(x3, y3);
    ctx.lineTo(x4, y4);
    ctx.closePath();
    ctx.fill();
  }

  function drawSegment(seg) {
    const th = game.theme;
    const c = seg.dark ? 1 : 0;
    const { x: x1, y: y1, w: w1 } = seg.p1.screen;
    const { x: x2, y: y2, w: w2 } = seg.p2.screen;
    const r1 = w1 / Math.max(6, 2 * LANES), r2 = w2 / Math.max(6, 2 * LANES);

    ctx.fillStyle = th.ground[c];
    ctx.fillRect(0, y2, W, y1 - y2);
    quad(th.rumble[c], x1 - w1 - r1, y1, x1 - w1, y1, x2 - w2, y2, x2 - w2 - r2, y2);
    quad(th.rumble[c], x1 + w1 + r1, y1, x1 + w1, y1, x2 + w2, y2, x2 + w2 + r2, y2);
    quad(seg.line === 'white' ? '#f2f2f2' : th.road[c], x1 - w1, y1, x1 + w1, y1, x2 + w2, y2, x2 - w2, y2);

    if (seg.line === 'checkered') {
      const n = 10;
      for (let i = 0; i < n; i++) {
        if (i % 2) continue;
        const a1 = x1 - w1 + (2 * w1 * i) / n, b1 = x1 - w1 + (2 * w1 * (i + 1)) / n;
        const a2 = x2 - w2 + (2 * w2 * i) / n, b2 = x2 - w2 + (2 * w2 * (i + 1)) / n;
        quad('#f2f2f2', a1, y1, b1, y1, b2, y2, a2, y2);
      }
    } else if (!seg.dark && th.lane && !seg.line) {
      const l1 = w1 / Math.max(32, 8 * LANES), l2 = w2 / Math.max(32, 8 * LANES);
      const lw1 = (w1 * 2) / LANES, lw2 = (w2 * 2) / LANES;
      let lx1 = x1 - w1 + lw1, lx2 = x2 - w2 + lw2;
      for (let lane = 1; lane < LANES; lane++, lx1 += lw1, lx2 += lw2) {
        quad(th.lane, lx1 - l1 / 2, y1, lx1 + l1 / 2, y1, lx2 + l2 / 2, y2, lx2 - l2 / 2, y2);
      }
    }

    if (seg.fog < 1) {
      ctx.globalAlpha = 1 - seg.fog;
      ctx.fillStyle = th.fog;
      ctx.fillRect(0, y2, W, y1 - y2);
      ctx.globalAlpha = 1;
    }
  }

  // Draws a sprite with its anchor at (x, y), clipped below clipY (hill crests).
  function drawSprite(spr, scale, x, y, offsetX, offsetY, clipY, alpha = 1) {
    const destW = spr.worldW * scale * (W / 2);
    const destH = destW * (spr.canvas.height / spr.canvas.width);
    const dx = x + destW * offsetX;
    const dy = y + destH * offsetY;
    const clipH = clipY ? Math.max(0, dy + destH - clipY) : 0;
    if (clipH >= destH || destW < 0.5) return null;
    ctx.globalAlpha = alpha;
    ctx.drawImage(spr.canvas, 0, 0, spr.canvas.width, spr.canvas.height - (spr.canvas.height * clipH) / destH,
      dx, dy, destW, destH - clipH);
    ctx.globalAlpha = 1;
    return { x: dx, y: dy, w: destW, h: destH };
  }

  function tailLightGlow(rect) {
    if (!rect || !game.theme.night) return;
    ctx.globalCompositeOperation = 'lighter';
    const r = Math.max(1.5, rect.w * 0.09);
    for (const fx of [0.17, 0.83]) {
      const gx = rect.x + rect.w * fx, gy = rect.y + rect.h * 0.57;
      const grad = ctx.createRadialGradient(gx, gy, 0, gx, gy, r * 2);
      grad.addColorStop(0, 'rgba(255,60,40,0.9)');
      grad.addColorStop(1, 'rgba(255,0,0,0)');
      ctx.fillStyle = grad;
      ctx.fillRect(gx - r * 2, gy - r * 2, r * 4, r * 4);
    }
    ctx.globalCompositeOperation = 'source-over';
  }

  function drawLayer(layer, rotation, dy) {
    const lw = layer.width;
    const sx = Math.floor((((rotation % 1) + 1) % 1) * lw);
    const first = Math.min(W, lw - sx);
    ctx.drawImage(layer, sx, 0, first, H, 0, dy, first, H);
    if (first < W) ctx.drawImage(layer, 0, 0, W - first, H, first, dy, W - first, H);
  }

  function render() {
    const t = game.track;
    const th = game.theme;
    const segs = t.segments;
    const bg = getBackground(game.stage.theme);
    const base = t.find(game.position);
    const basePct = (game.position % SEG) / SEG;
    const playerSeg = t.find(game.position + PLAYER_Z);
    const playerPct = ((game.position + PLAYER_Z) % SEG) / SEG;
    const playerY = lerp(playerSeg.p1.world.y, playerSeg.p2.world.y, playerPct);

    ctx.save();
    if (game.shake > 0) ctx.translate((Math.random() - 0.5) * 6 * game.shake / 0.3, (Math.random() - 0.5) * 4 * game.shake / 0.3);

    ctx.drawImage(bg.sky, 0, 0);
    const lift = clamp(-playerY * 0.0006, -18, 18);
    drawLayer(bg.far, game.skyOffset, lift * 0.5);
    drawLayer(bg.near, game.nearOffset, lift);

    let maxy = H;
    let x = 0;
    let dx = -(base.curve * basePct);
    const visible = Math.min(DRAW_DISTANCE, segs.length - base.index);

    for (let n = 0; n < visible; n++) {
      const seg = segs[base.index + n];
      seg.fog = fogAt(n / DRAW_DISTANCE, th.fogDensity);
      seg.clip = maxy;
      project(seg.p1, game.playerX * ROAD_WIDTH - x, playerY + CAMERA_HEIGHT, game.position);
      project(seg.p2, game.playerX * ROAD_WIDTH - x - dx, playerY + CAMERA_HEIGHT, game.position);
      x += dx;
      dx += seg.curve;
      seg.visible = !(seg.p1.camera.z <= CAMERA_DEPTH || seg.p2.screen.y >= seg.p1.screen.y || seg.p2.screen.y >= maxy);
      if (!seg.visible) continue;
      drawSegment(seg);
      maxy = seg.p1.screen.y;
    }

    for (let n = visible - 1; n > 0; n--) {
      const seg = segs[base.index + n];
      if (seg.p1.camera.z > CAMERA_DEPTH) {
        const fade = th.fogDensity > 10 ? clamp(seg.fog * 1.2, 0, 1) : 1;
        for (const car of seg.cars) {
          const scale = lerp(seg.p1.screen.scale, seg.p2.screen.scale, car.percent);
          const sx = lerp(seg.p1.screen.x, seg.p2.screen.x, car.percent) + (scale * car.offset * ROAD_WIDTH * W) / 2;
          const sy = lerp(seg.p1.screen.y, seg.p2.screen.y, car.percent);
          tailLightGlow(drawSprite(car.sprite, scale, sx, sy, -0.5, -1, seg.clip, fade));
        }
        for (const s of seg.sprites) {
          const spr = SPRITES[s.name];
          const scale = seg.p1.screen.scale;
          const sx = seg.p1.screen.x + (scale * s.offset * ROAD_WIDTH * W) / 2;
          drawSprite(spr, scale, sx, seg.p1.screen.y, s.center ? -0.5 : s.offset < 0 ? -1 : 0, -1, seg.clip, fade);
        }
      }
      if (seg === playerSeg) drawPlayer(playerSeg, playerPct);
    }

    if (th.night) {
      const grad = ctx.createRadialGradient(W / 2, H * 0.95, 30, W / 2, H * 0.7, W * 0.75);
      grad.addColorStop(0, 'rgba(0,0,8,0)');
      grad.addColorStop(1, 'rgba(0,0,8,0.55)');
      ctx.fillStyle = grad;
      ctx.fillRect(0, 0, W, H);
    }
    drawWeather();
    ctx.restore();
  }

  function drawPlayer(playerSeg, playerPct) {
    const scale = CAMERA_DEPTH / PLAYER_Z;
    const camY = lerp(playerSeg.p1.camera.y, playerSeg.p2.camera.y, playerPct);
    const speedPct = game.speed / MAX_SPEED;
    const bounce = speedPct > 0 ? (Math.random() < 0.5 ? -1 : 1) * Math.random() * speedPct * 1.2 : 0;
    const y = H / 2 - (scale * camY * H) / 2 + bounce;
    const spr = game.steer < 0 ? PLAYER_SPRITES.left : game.steer > 0 ? PLAYER_SPRITES.right : PLAYER_SPRITES.straight;
    tailLightGlow(drawSprite(spr, scale, W / 2, y, -0.5, -1));
  }

  function drawWeather() {
    const kind = game.theme.weather;
    if (!kind) return;
    if (kind === 'snow') {
      ctx.fillStyle = 'rgba(255,255,255,0.9)';
      for (const p of particles) {
        const s = p.z > 0.7 ? 2 : 1;
        ctx.fillRect(p.x, p.y, s, s);
      }
    } else {
      ctx.strokeStyle = 'rgba(190,205,230,0.45)';
      ctx.lineWidth = 1;
      ctx.beginPath();
      for (const p of particles) {
        ctx.moveTo(p.x, p.y);
        ctx.lineTo(p.x - 2, p.y + 6 + p.z * 6);
      }
      ctx.stroke();
    }
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
    const { gear, rpm } = gearbox();
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
        + (progress.best ? `<p class="stats">BEST SCORE ${progress.best}</p>` : ''),
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
    render();
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

  startAttract();
  showTitle();
  requestAnimationFrame((t) => { last = t; frame(t); });
})();
