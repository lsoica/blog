(() => {
  'use strict';

  // ===========================================================================
  // Turbo Rally: AI Academy. A population of small neural networks learns to
  // drive by neuroevolution, using the exact physics of the game (shared
  // through ../turbo-rally/engine.js).
  //
  // To make this real learning rather than memorisation, every generation
  // races on a brand-new random track with slow traffic, the score is the
  // average speed over a fixed 40-second race, a simple hand-written rule
  // races alongside as a reference, and the champion is picked on held-out
  // test tracks the population never trains on.
  // ===========================================================================

  const E = window.TurboEngine;
  const {
    SEG, PLAYER_Z, STEP, MAX_SPEED, TOP_KMH, DRAW_DISTANCE, STAGES, THEMES, CAR_W,
    clamp, mulberry32, buildTrack, stepCar, carSprites, PLAYER_SPRITES, RIVAL_SPRITES,
  } = E;

  // ---------------------------------------------------------------------------
  // The driver: an 11-10-2 neural network
  // ---------------------------------------------------------------------------

  // Test-only overrides (?eyes=nearest|lanes&nh=N) for comparing designs.
  const params = new URLSearchParams(location.search);
  const EYES = params.get('eyes') || 'nearest';

  const LOOKAHEAD = [4, 12, 24, 40, 64]; // segments ahead at which bends are sensed
  const SIGHT = 50;                       // segments ahead at which traffic is seen
  // "lanes" eyes split the road ahead into five strips relative to the car and
  // report how close the nearest car in each strip is, so gaps are visible.
  const LANE_EDGES = [-0.75, -0.25, 0.25, 0.75];
  const TRAFFIC_LABELS = EYES === 'lanes'
    ? ['Far left', 'Left', 'Straight ahead', 'Right', 'Far right']
    : ['Car ahead', 'Car side', 'Car in path'];
  const INPUT_LABELS = ['Position', 'Speed', ...LOOKAHEAD.map((n) => `Bend +${n}`), 'Hill', ...TRAFFIC_LABELS];
  const TRAFFIC_INPUTS = TRAFFIC_LABELS.map((_, i) => 8 + i);
  const OUTPUT_LABELS = ['Steer', 'Pedals'];
  const NI = INPUT_LABELS.length;
  const NH = Number(params.get('nh')) || 10;
  const NO = OUTPUT_LABELS.length;
  // Weights are stored flat: input->hidden, hidden biases, hidden->output, output biases.
  const W2 = NH * NI + NH;
  const GENOME = W2 + NO * NH + NO;

  function gaussian() {
    let u = 0, v = 0;
    while (!u) u = Math.random();
    while (!v) v = Math.random();
    return Math.sqrt(-2 * Math.log(u)) * Math.cos(2 * Math.PI * v);
  }

  const randomGenome = () => Float32Array.from({ length: GENOME }, () => gaussian() * 0.8);

  function forward(g, inputs, hidden, out) {
    for (let h = 0; h < NH; h++) {
      let s = g[NH * NI + h];
      for (let i = 0; i < NI; i++) s += g[h * NI + i] * inputs[i];
      hidden[h] = Math.tanh(s);
    }
    for (let o = 0; o < NO; o++) {
      let s = g[W2 + NO * NH + o];
      for (let h = 0; h < NH; h++) s += g[W2 + o * NH + h] * hidden[h];
      out[o] = Math.tanh(s);
    }
  }

  // What a car "sees", each roughly in the range -1..1.
  function sense(car, track, inputs) {
    const segs = track.segments;
    const seg = track.find(car.position + PLAYER_Z);
    const at = (n) => segs[Math.min(segs.length - 1, seg.index + n)];
    inputs[0] = car.playerX / 2.5;
    inputs[1] = car.speed / MAX_SPEED;
    for (let i = 0; i < LOOKAHEAD.length; i++) inputs[2 + i] = at(LOOKAHEAD[i]).curve / 6;
    inputs[7] = clamp(((at(30).p1.world.y - seg.p1.world.y) / (SEG * 30)) * 3, -1, 1);

    const z = car.position + PLAYER_Z;
    if (EYES === 'lanes') {
      for (let k = 0; k < 5; k++) inputs[8 + k] = 0;
      for (let i = 0; i < SIGHT; i++) {
        for (const other of at(i).cars) {
          if (other.z < z) continue;
          const dx = other.offset - car.playerX;
          let lane = 0;
          while (lane < LANE_EDGES.length && dx >= LANE_EDGES[lane]) lane++;
          if (!inputs[8 + lane]) inputs[8 + lane] = 1 - (other.z - z) / (SIGHT * SEG);
        }
      }
      return;
    }

    // Traffic: the nearest car ahead, and the nearest one directly in our path.
    let near = 0, side = 0, path = 0;
    for (let i = 0; i < SIGHT && !(near && path); i++) {
      for (const other of at(i).cars) {
        if (other.z < z) continue;
        const closeness = 1 - (other.z - z) / (SIGHT * SEG);
        if (!near) {
          near = closeness;
          side = clamp((other.offset - car.playerX) / 1.5, -1, 1);
        }
        if (!path && Math.abs(other.offset - car.playerX) < CAR_W * 1.6) path = closeness;
      }
    }
    inputs[8] = near;
    inputs[9] = side;
    inputs[10] = path;
  }

  const controlsFrom = (out) => ({
    left: out[0] < -0.25,
    right: out[0] > 0.25,
    up: out[1] > 0,
    down: out[1] < -0.6,
  });

  // The reference driver: flat out, steering back towards the middle of the
  // road. With no traffic this is already the fastest way round every track.
  const simpleRule = (car) => ({ left: car.playerX > 0.05, right: car.playerX < -0.05, up: true, down: false });

  // ---------------------------------------------------------------------------
  // Tracks and traffic
  // ---------------------------------------------------------------------------

  const RACE_SECONDS = 40;      // every race lasts this long; the score is distance covered
  const TRACK_SEGMENTS = 2600;  // longer than anyone can drive in 40 seconds
  const TRAFFIC = 36;
  const TRAFFIC_SPEED = [0.35, 0.65]; // fraction of top speed
  // Traffic in every colour except the blues, which would look like learners.
  const TRAFFIC_SPRITES = RIVAL_SPRITES.filter((_, i) => i !== 0 && i !== 7);
  const SPREAD = 0.75;              // traffic uses the whole road, not fixed lanes
  const WEAVE_SPEED = 0.35;         // how fast traffic drifts sideways (road half-widths per second)
  const WEAVE_EVERY = [2, 6];       // seconds between drifts to a new position
  const TEST_SEEDS = [7001, 7002, 7003];

  // A random track in the style (scenery, bends, hills, grip) of one stage.
  function makeTrack(styleIndex, seed, segments = TRACK_SEGMENTS) {
    const style = STAGES[styleIndex];
    const track = buildTrack({ ...style, seed, length: segments, checkpoints: 1 });
    track.traffic = makeTraffic(track, mulberry32(seed * 7 + 3), TRAFFIC * (track.finish / TRACK_SEGMENTS));
    return track;
  }

  // Traffic spreads across the road and drifts sideways at random moments, so
  // no fixed line is ever safe: drivers have to watch the cars ahead.
  function makeTraffic(track, rng, count) {
    const cars = [];
    const first = 60, last = track.finish - 40;
    track.weaveRng = rng;
    for (let k = 0; k < count; k++) {
      const z = (first + ((last - first) * (k + rng())) / count) * SEG;
      const offset = (rng() * 2 - 1) * SPREAD;
      const car = {
        z,
        offset,
        target: offset,
        retarget: WEAVE_EVERY[0] + rng() * (WEAVE_EVERY[1] - WEAVE_EVERY[0]),
        speed: MAX_SPEED * (TRAFFIC_SPEED[0] + rng() * (TRAFFIC_SPEED[1] - TRAFFIC_SPEED[0])),
        sprite: TRAFFIC_SPRITES[k % TRAFFIC_SPRITES.length],
        percent: 0,
      };
      track.find(z).cars.push(car);
      cars.push(car);
    }
    return cars;
  }

  // Traffic keeps its speed and weaves; cars leave at the end of the road.
  function moveTraffic(track, dt) {
    const end = (track.segments.length - DRAW_DISTANCE - 20) * SEG;
    const rng = track.weaveRng;
    for (let i = track.traffic.length - 1; i >= 0; i--) {
      const car = track.traffic[i];
      car.retarget -= dt;
      if (car.retarget <= 0) {
        car.target = (rng() * 2 - 1) * SPREAD;
        car.retarget = WEAVE_EVERY[0] + rng() * (WEAVE_EVERY[1] - WEAVE_EVERY[0]);
      }
      car.offset += clamp(car.target - car.offset, -WEAVE_SPEED * dt, WEAVE_SPEED * dt);
      const oldSeg = track.find(car.z);
      car.z += dt * car.speed;
      car.percent = (car.z % SEG) / SEG;
      const newSeg = track.find(car.z);
      if (car.z >= end) {
        oldSeg.cars.splice(oldSeg.cars.indexOf(car), 1);
        track.traffic.splice(i, 1);
      } else if (newSeg !== oldSeg) {
        oldSeg.cars.splice(oldSeg.cars.indexOf(car), 1);
        newSeg.cars.push(car);
      }
    }
  }

  // ---------------------------------------------------------------------------
  // Evolution
  // ---------------------------------------------------------------------------

  const POPULATION = 40;
  const ELITES = 4;          // best drivers copied unchanged into the next generation
  const IMMIGRANTS = 2;      // brand-new random drivers each generation, to keep variety
  const TOURNAMENT = 3;      // parents are the best of 3 random picks from the top half
  const MUTATION_RATE = 0.1; // chance that each weight of a child is nudged
  const MUTATION_SIZE = 0.35;

  // A car is retired early when it is stuck, off the road too long, or not getting anywhere.
  const STUCK_SPEED = MAX_SPEED * 0.08;
  const STUCK_SECONDS = 2;
  const OFFROAD_SECONDS = 4;
  const PROGRESS_WINDOW = 5; // seconds
  const PROGRESS_MIN = 10;   // segments that must be covered in each window

  function tournament(ranked) {
    let best = Infinity;
    for (let i = 0; i < TOURNAMENT; i++) best = Math.min(best, Math.floor(Math.random() * (ranked.length / 2)));
    return ranked[best].genome;
  }

  function breed(a, b, size) {
    const child = new Float32Array(GENOME);
    for (let i = 0; i < GENOME; i++) {
      child[i] = Math.random() < 0.5 ? a[i] : b[i];
      if (Math.random() < MUTATION_RATE) child[i] += gaussian() * size;
    }
    return child;
  }

  // `stagnant` counts generations without a new champion; mutations grow with
  // it so a population stuck on one bad habit gets shaken out of it.
  function nextGeneration(ranked, stagnant) {
    const size = MUTATION_SIZE * (1 + Math.min(stagnant, 10) * 0.2);
    const next = ranked.slice(0, ELITES).map((c) => c.genome.slice());
    for (let i = 0; i < IMMIGRANTS; i++) next.push(randomGenome());
    while (next.length < POPULATION) next.push(breed(tournament(ranked), tournament(ranked), size));
    return next;
  }

  // ---------------------------------------------------------------------------
  // Racing
  // ---------------------------------------------------------------------------

  const kmh = (segments, seconds) => (segments * SEG) / seconds / MAX_SPEED * TOP_KMH;

  function newCar(genome) {
    return {
      genome,
      position: 0, playerX: 0, speed: 0, steer: 0, throttle: 0,
      alive: true, finished: false, finishTime: 0,
      crashes: 0, touching: false, offroad: 0, stuck: 0, checkTime: 0, checkPos: 0,
      inputs: new Float32Array(NI), hidden: new Float32Array(NH), out: new Float32Array(NO),
    };
  }

  const distance = (car, track) => Math.min(track.finish, (car.position + PLAYER_Z) / SEG);
  const BUMP_PENALTY = Number(params.get('bump')) || 25; // segments of score lost per collision
  const fitness = (car, track) => distance(car, track) - car.crashes * BUMP_PENALTY;

  let inputMask = null; // debug only: zeroes chosen inputs to test what a driver relies on

  // Advances one car, applying the retirement rules. Returns true while it races.
  // `policy` is null for network drivers, or a function returning controls.
  function driveCar(car, track, theme, dt, time, limit, policy = null) {
    let controls;
    if (policy) {
      controls = policy(car, track);
    } else {
      sense(car, track, car.inputs);
      if (inputMask) for (let i = 0; i < NI; i++) if (!inputMask[i]) car.inputs[i] = 0;
      forward(car.genome, car.inputs, car.hidden, car.out);
      controls = controlsFrom(car.out);
    }
    const hit = stepCar(car, controls, track, theme, dt) !== 0;
    if (hit && !car.touching) car.crashes++;
    car.touching = hit;

    if (car.position + PLAYER_Z >= track.finish * SEG) {
      car.finished = true;
      car.finishTime = time;
      return false;
    }
    car.offroad = Math.abs(car.playerX) > 1 ? car.offroad + dt : 0;
    car.stuck = time > 3 && car.speed < STUCK_SPEED ? car.stuck + dt : 0;
    if (car.offroad > OFFROAD_SECONDS || car.stuck > STUCK_SECONDS || time >= limit) return false;
    if (time - car.checkTime >= PROGRESS_WINDOW) {
      if ((car.position - car.checkPos) / SEG < PROGRESS_MIN) return false;
      car.checkTime = time;
      car.checkPos = car.position;
    }
    return true;
  }

  // Races one driver alone on a fresh copy of a track (same layout and traffic
  // every time for a given seed). Returns distance, km/h and crashes.
  function solo(genomeOrPolicy, styleIndex, seed, { segments, limit = RACE_SECONDS } = {}) {
    const track = segments ? makeTrack(styleIndex, seed, segments) : makeTrack(styleIndex, seed);
    const theme = THEMES[STAGES[styleIndex].theme];
    const policy = typeof genomeOrPolicy === 'function' ? genomeOrPolicy : null;
    const car = newCar(policy ? null : genomeOrPolicy);
    let t = 0;
    do {
      t += STEP;
      moveTraffic(track, STEP);
    } while (driveCar(car, track, theme, STEP, t, limit, policy));
    const dist = distance(car, track);
    return { dist, kmh: kmh(dist, limit), crashes: car.crashes, finished: car.finished, time: car.finished ? car.finishTime : t };
  }

  const MIXED = -1; // track style setting: a random style every generation
  const styleName = (sel) => (sel === MIXED ? 'mixed' : STAGES[sel].name);

  // Average km/h over the held-out test tracks: three of the chosen style, or
  // one of each style when training on mixed tracks.
  function testScore(genomeOrPolicy, sel) {
    const plan = sel === MIXED
      ? STAGES.map((_, i) => [i, TEST_SEEDS[i % TEST_SEEDS.length] + i])
      : TEST_SEEDS.map((seed) => [sel, seed]);
    const runs = plan.map(([style, seed]) => solo(genomeOrPolicy, style, seed));
    return {
      kmh: runs.reduce((s, r) => s + r.kmh, 0) / runs.length,
      crashes: runs.reduce((s, r) => s + r.crashes, 0),
    };
  }

  // ---------------------------------------------------------------------------
  // State
  // ---------------------------------------------------------------------------

  const SAVE_KEY = 'turbo-rally-academy-v2';
  const GHOST_SPRITE = carSprites('#58d6ff').straight;
  const MAX_GHOSTS = 6; // other learners drawn ahead of the red car
  const $ = (id) => document.getElementById(id);

  const lab = {
    styleIndex: MIXED,  // the track style setting (MIXED or a stage index)
    genStyle: 1,        // the style of the current generation's track
    track: null,
    theme: null,
    generation: 1,
    population: [],
    cars: [],
    reference: null,    // the simple rule, racing the same track and traffic
    time: 0,
    running: true,
    speed: 4,
    history: [],        // [{ generation, best, avg, ref, styleIndex }] in km/h
    champion: null,     // { genome, test, styleIndex, generation }
    ruleTest: {},       // simple rule's test score per style
    stagnant: 0,
    mode: 'train',      // train | watch
    watch: null,
    follow: 0,
    cameraMode: 'best', // best: last generation's winner; pack: middle of the field; leader: front car
    camera: { last: 0, skyOffset: 0, nearOffset: 0 },
  };

  function setStyle(sel) {
    lab.styleIndex = sel;
    $('stage').value = String(sel);
    if (!lab.ruleTest[sel]) lab.ruleTest[sel] = testScore(simpleRule, sel);
  }

  function startGeneration() {
    lab.genStyle = lab.styleIndex === MIXED ? Math.floor(Math.random() * STAGES.length) : lab.styleIndex;
    lab.theme = THEMES[STAGES[lab.genStyle].theme];
    E.resetWeather(lab.theme);
    lab.track = makeTrack(lab.genStyle, (Math.random() * 1e9) | 0);
    lab.cars = lab.population.map(newCar);
    lab.reference = newCar(null);
    lab.time = 0;
    lab.follow = 0;
    lab.camera.last = 0;
    if (lab.speed !== 'max') banner(`GENERATION ${lab.generation}<small>a brand-new ${STAGES[lab.genStyle].name.toLowerCase()} track</small>`, 1.4);
    updateStats();
  }

  function stepGeneration(dt) {
    lab.time += dt;
    moveTraffic(lab.track, dt);
    let racing = 0;
    for (const car of lab.cars) {
      if (!car.alive) continue;
      car.alive = driveCar(car, lab.track, lab.theme, dt, lab.time, RACE_SECONDS);
      if (car.alive) racing++;
    }
    const ref = lab.reference;
    if (ref.alive) ref.alive = driveCar(ref, lab.track, lab.theme, dt, lab.time, RACE_SECONDS, simpleRule);
    if (racing === 0 && !ref.alive) endGeneration();
  }

  function endGeneration() {
    const t = lab.track;
    const ranked = lab.cars
      .map((car) => ({ fitness: fitness(car, t), dist: distance(car, t), genome: car.genome }))
      .sort((a, b) => b.fitness - a.fitness);
    const best = kmh(ranked[0].dist, RACE_SECONDS);
    const avg = ranked.reduce((s, r) => s + kmh(r.dist, RACE_SECONDS), 0) / ranked.length;
    const ref = kmh(distance(lab.reference, t), RACE_SECONDS);
    lab.history.push({ generation: lab.generation, best, avg, ref, styleIndex: lab.styleIndex });

    // The champion is decided on the held-out test tracks, not on this one race,
    // so a driver that just got lucky with this layout doesn't win the title.
    const test = testScore(ranked[0].genome, lab.styleIndex);
    const champ = lab.champion;
    const improved = !champ || champ.styleIndex !== lab.styleIndex || test.kmh > champ.test.kmh + 0.5;
    lab.stagnant = improved ? 0 : lab.stagnant + 1;
    if (improved) {
      lab.champion = { genome: ranked[0].genome.slice(), test, styleIndex: lab.styleIndex, generation: lab.generation };
    }
    if (lab.speed !== 'max') {
      banner(`GEN ${lab.generation} DONE<small>best ${Math.round(best)} km/h · simple rule ${Math.round(ref)} km/h</small>`, 1.6);
    }

    lab.population = nextGeneration(ranked, lab.stagnant);
    lab.generation++;
    save();
    drawChart();
    startGeneration();
  }

  // ---------------------------------------------------------------------------
  // Watching the champion on the real stage
  // ---------------------------------------------------------------------------

  const WATCH_LIMIT = 240;

  function realStage(index) {
    const track = buildTrack(STAGES[index]);
    track.traffic = makeTraffic(track, mulberry32(STAGES[index].seed + 5), 19);
    return track;
  }

  function startWatch() {
    if (!lab.champion) return;
    // Race the real stage of the style currently on screen. Time the simple
    // rule on the identical stage first, for comparison.
    const stage = lab.genStyle;
    const theme = THEMES[STAGES[stage].theme];
    const ruleTrack = realStage(stage);
    const rule = newCar(null);
    let t = 0;
    do { t += STEP; moveTraffic(ruleTrack, STEP); } while (driveCar(rule, ruleTrack, theme, STEP, t, WATCH_LIMIT, simpleRule));

    lab.mode = 'watch';
    lab.watch = {
      stage,
      theme,
      track: realStage(stage),
      car: newCar(lab.champion.genome),
      time: 0,
      done: 0,
      rule: rule.finished ? rule.finishTime : null,
    };
    lab.camera.last = 0;
    $('watch').textContent = '■ Stop watching';
    banner(`CHAMPION<small>from generation ${lab.champion.generation}, on the real ${STAGES[stage].name} stage</small>`, 1.8);
  }

  function stopWatch() {
    lab.mode = 'train';
    lab.watch = null;
    E.resetWeather(lab.theme);
    lab.camera.last = lab.cars[lab.follow] ? lab.cars[lab.follow].position : 0;
    $('watch').textContent = '▶ Watch champion';
    banner('', 0);
  }

  function stepWatch(dt) {
    const w = lab.watch;
    if (w.done) {
      w.done += dt;
      if (w.done > 4) stopWatch();
      return;
    }
    w.time += dt;
    moveTraffic(w.track, dt);
    w.car.alive = driveCar(w.car, w.track, w.theme, dt, w.time, WATCH_LIMIT);
    if (!w.car.alive) {
      w.done = 0.001;
      const rule = w.rule ? `simple rule: ${w.rule.toFixed(1)}s` : 'the simple rule did not finish';
      banner(w.car.finished
        ? `FINISHED IN ${w.car.finishTime.toFixed(1)}s<small>${rule}</small>`
        : `RETIRED AT ${Math.round((distance(w.car, w.track) / w.track.finish) * 100)}%<small>${rule}</small>`, 4);
    }
  }

  // ---------------------------------------------------------------------------
  // Saving
  // ---------------------------------------------------------------------------

  const pack = (g) => Array.from(g, (x) => Math.round(x * 1e4) / 1e4);

  function save() {
    try {
      localStorage.setItem(SAVE_KEY, JSON.stringify({
        genome: GENOME,
        styleIndex: lab.styleIndex, generation: lab.generation,
        population: lab.population.map(pack), history: lab.history,
        champion: lab.champion && { ...lab.champion, genome: pack(lab.champion.genome) },
      }));
    } catch { /* storage full or disabled: training still works, it just won't persist */ }
  }

  function restore() {
    try {
      const data = JSON.parse(localStorage.getItem(SAVE_KEY));
      if (!data || data.genome !== GENOME) return false;
      lab.generation = data.generation;
      lab.population = data.population.map((g) => Float32Array.from(g));
      lab.history = data.history || [];
      lab.champion = data.champion && { ...data.champion, genome: Float32Array.from(data.champion.genome) };
      setStyle(clamp(data.styleIndex | 0, MIXED, STAGES.length - 1));
      return true;
    } catch {
      return false;
    }
  }

  function resetLab() {
    lab.generation = 1;
    lab.population = Array.from({ length: POPULATION }, randomGenome);
    lab.history = [];
    lab.champion = null;
    lab.stagnant = 0;
    if (lab.mode === 'watch') stopWatch();
    save();
    drawChart();
    startGeneration();
  }

  // ---------------------------------------------------------------------------
  // Rendering: race view, HUD, chart and network diagram
  // ---------------------------------------------------------------------------

  // Picks the car the camera rides with. "Best" is last generation's winner,
  // which always races again unchanged (car 0). "Pack" stays with a car in the
  // middle of the field so the cars ahead of it are on screen; "leader" follows
  // the front car. Pack and leader only switch when they have to.
  function followedCar() {
    if (lab.mode === 'watch') return lab.watch.car;
    const current = lab.cars[lab.follow];
    if (lab.cameraMode === 'best') {
      lab.follow = 0;
    } else if (lab.cameraMode === 'leader') {
      let lead = 0;
      lab.cars.forEach((c, i) => { if (c.position > lab.cars[lead].position) lead = i; });
      if (lead !== lab.follow && (!current.alive || lab.cars[lead].position - current.position > SEG * 3)) lab.follow = lead;
    } else {
      const racing = lab.cars.filter((c) => c.alive).sort((a, b) => b.position - a.position);
      const rank = racing.indexOf(current);
      const n = racing.length;
      if (n && (rank < 0 || rank < n * 0.2 - 1 || rank > n * 0.8)) lab.follow = lab.cars.indexOf(racing[Math.floor(n / 2)]);
    }
    return lab.cars[lab.follow];
  }

  function renderRace(dt) {
    const car = followedCar();
    if (!car) return;
    const watching = lab.mode === 'watch';
    const track = watching ? lab.watch.track : lab.track;
    const theme = watching ? lab.watch.theme : lab.theme;
    const style = watching ? lab.watch.stage : lab.genStyle;
    const seg = track.find(car.position + PLAYER_Z);
    const moved = car.position - lab.camera.last;
    if (Math.abs(moved) < SEG * 20) {
      lab.camera.skyOffset = (lab.camera.skyOffset + 0.0012 * seg.curve * (moved / SEG) + 1) % 1;
      lab.camera.nearOffset = (lab.camera.nearOffset + 0.0024 * seg.curve * (moved / SEG) + 1) % 1;
    }
    lab.camera.last = car.position;
    E.updateWeather(dt, theme, seg.curve, car.speed / MAX_SPEED, car.steer);

    // Only draw learners clearly ahead of the red car. Everyone starts on the
    // same spot, and cars level with the camera would be drawn huge and stack
    // into a smeared band across the bottom of the screen.
    // Learners with near-identical networks drive the same line side by side;
    // drawing each would show a row of see-through copies, so cars sharing a
    // spot are merged into one, drawn more solid the more cars it stands for.
    let ghosts = null;
    if (lab.mode === 'train') {
      const spots = new Map();
      for (const c of lab.cars) {
        if (c === car || c.position - car.position <= SEG * 2) continue;
        const key = `${Math.floor(c.position / (SEG * 2))}:${Math.round(c.playerX / 0.3)}:${c.alive}`;
        const spot = spots.get(key);
        if (spot) spot.count++;
        else spots.set(key, { z: c.position + PLAYER_Z, offset: c.playerX, sprite: GHOST_SPRITE, alive: c.alive, count: 1 });
      }
      // Learners pass through each other, so a bunch driving abreast overlaps
      // into a row of see-through cars. Show only the nearest few ahead.
      ghosts = [...spots.values()]
        .filter((g) => g.alive)
        .sort((a, b) => a.z - b.z)
        .slice(0, MAX_GHOSTS)
        .map((g) => ({ ...g, alpha: Math.min(0.75, 0.4 + 0.07 * (g.count - 1)) }));
    }

    E.render({
      track, theme, themeName: STAGES[style].theme,
      position: car.position, playerX: car.playerX, speed: car.speed, steer: car.steer,
      skyOffset: lab.camera.skyOffset, nearOffset: lab.camera.nearOffset, shake: 0,
      ghosts, playerSprites: PLAYER_SPRITES,
    });
    drawNet(car);
  }

  const setText = (id, text) => { const el = $(id); if (el.textContent !== text) el.textContent = text; };

  function updateHud() {
    const car = followedCar();
    if (lab.mode === 'watch') {
      setText('hud-gen', 'CHAMP');
      setText('hud-alive', '1/1');
      setText('hud-time', lab.watch.time.toFixed(1));
    } else {
      setText('hud-gen', String(lab.generation));
      setText('hud-alive', `${lab.cars.filter((c) => c.alive).length}/${POPULATION}`);
      setText('hud-time', lab.time.toFixed(1));
    }
    if (car) {
      setText('hud-speed', `${Math.round((car.speed / MAX_SPEED) * TOP_KMH)}`);
      setText('hud-bumps', String(car.crashes));
    }
  }

  function updateStats() {
    setText('stat-gen', String(lab.generation));
    const last = lab.history[lab.history.length - 1];
    setText('stat-best', last ? `${Math.round(last.best)} km/h` : '–');
    const c = lab.champion && lab.champion.styleIndex === lab.styleIndex ? lab.champion : null;
    setText('stat-champ', c ? `${Math.round(c.test.kmh)} km/h` : '–');
    setText('stat-rule', `${Math.round(lab.ruleTest[lab.styleIndex].kmh)} km/h`);
    $('watch').disabled = !lab.champion;
  }

  function sizeCanvas(cv) {
    const dpr = window.devicePixelRatio || 1;
    const w = cv.clientWidth, h = cv.clientHeight;
    if (cv.width !== Math.round(w * dpr) || cv.height !== Math.round(h * dpr)) {
      cv.width = Math.round(w * dpr);
      cv.height = Math.round(h * dpr);
    }
    const g = cv.getContext('2d');
    g.setTransform(dpr, 0, 0, dpr, 0, 0);
    return { g, w, h };
  }

  function drawChart() {
    const { g, w, h } = sizeCanvas($('chart'));
    g.clearRect(0, 0, w, h);
    const pad = { l: 30, r: 8, t: 8, b: 18 };
    const cw = w - pad.l - pad.r, ch = h - pad.t - pad.b;
    const top = TOP_KMH;
    g.font = '10px system-ui, sans-serif';
    g.fillStyle = '#6b7088';
    g.strokeStyle = 'rgba(255,255,255,0.08)';
    g.lineWidth = 1;
    for (const v of [0, 80, 160, 240]) {
      const y = pad.t + ch * (1 - v / top);
      g.beginPath(); g.moveTo(pad.l, y + 0.5); g.lineTo(w - pad.r, y + 0.5); g.stroke();
      g.textAlign = 'right';
      g.fillText(String(v), pad.l - 4, y + 3);
    }
    const hist = lab.history.slice(-120);
    if (hist.length === 0) {
      g.textAlign = 'center';
      g.fillText('The first generation is still racing…', pad.l + cw / 2, pad.t + ch / 2);
      return;
    }
    const x = (i) => pad.l + (hist.length === 1 ? cw / 2 : (i / (hist.length - 1)) * cw);
    const y = (v) => pad.t + ch * (1 - Math.min(v, top) / top);

    // Mark where the track style was switched.
    g.setLineDash([3, 3]);
    g.strokeStyle = 'rgba(255,255,255,0.25)';
    for (let i = 1; i < hist.length; i++) {
      if (hist[i].styleIndex !== hist[i - 1].styleIndex) {
        g.beginPath(); g.moveTo(x(i) + 0.5, pad.t); g.lineTo(x(i) + 0.5, pad.t + ch); g.stroke();
        g.textAlign = 'left';
        g.fillText(styleName(hist[i].styleIndex), x(i) + 3, pad.t + 9);
      }
    }
    g.setLineDash([]);

    const line = (key, color, width, dash = []) => {
      g.setLineDash(dash);
      g.strokeStyle = color;
      g.lineWidth = width;
      g.beginPath();
      hist.forEach((p, i) => (i ? g.lineTo(x(i), y(p[key])) : g.moveTo(x(i), y(p[key]))));
      g.stroke();
      g.setLineDash([]);
    };
    line('ref', 'rgba(255,255,255,0.7)', 1.2, [4, 3]);
    line('avg', '#6b7aa8', 1.5);
    line('best', '#ffd84a', 2);

    g.fillStyle = '#6b7088';
    g.textAlign = 'left';
    g.fillText(`gen ${hist[0].generation}`, pad.l, h - 4);
    g.textAlign = 'right';
    g.fillText(`gen ${hist[hist.length - 1].generation} · km/h`, w - pad.r, h - 4);
  }

  function drawNet(car) {
    if (!car.genome) return;
    const { g, w, h } = sizeCanvas($('net'));
    g.clearRect(0, 0, w, h);
    const colX = [68, w / 2, w - 70];
    const layerY = (n, i) => 12 + ((h - 24) * (i + 0.5)) / n;
    const nodes = [
      Array.from({ length: NI }, (_, i) => [colX[0], layerY(NI, i), car.inputs[i]]),
      Array.from({ length: NH }, (_, i) => [colX[1], layerY(NH, i), car.hidden[i]]),
      Array.from({ length: NO }, (_, i) => [colX[2], layerY(NO, i), car.out[i]]),
    ];
    const gw = car.genome;
    const edge = (a, b, weight) => {
      const m = Math.min(1, Math.abs(weight) / 2.5);
      g.strokeStyle = weight > 0 ? `rgba(74,163,255,${0.1 + m * 0.7})` : `rgba(255,122,74,${0.1 + m * 0.7})`;
      g.lineWidth = 0.4 + m * 1.6;
      g.beginPath(); g.moveTo(a[0], a[1]); g.lineTo(b[0], b[1]); g.stroke();
    };
    for (let hI = 0; hI < NH; hI++) for (let i = 0; i < NI; i++) edge(nodes[0][i], nodes[1][hI], gw[hI * NI + i]);
    for (let o = 0; o < NO; o++) for (let hI = 0; hI < NH; hI++) edge(nodes[1][hI], nodes[2][o], gw[W2 + o * NH + hI]);

    for (const layer of nodes) {
      for (const [nx, ny, v] of layer) {
        const a = Math.min(1, Math.abs(v));
        g.fillStyle = v >= 0 ? `rgba(120,190,255,${0.25 + a * 0.75})` : `rgba(255,150,100,${0.25 + a * 0.75})`;
        g.beginPath(); g.arc(nx, ny, 5, 0, Math.PI * 2); g.fill();
        g.strokeStyle = 'rgba(255,255,255,0.5)';
        g.lineWidth = 1;
        g.stroke();
      }
    }
    g.font = '10px system-ui, sans-serif';
    g.fillStyle = '#aab0d4';
    g.textAlign = 'right';
    nodes[0].forEach(([nx, ny], i) => g.fillText(INPUT_LABELS[i], nx - 9, ny + 3));
    g.textAlign = 'left';
    const c = controlsFrom(car.out);
    const decisions = [c.left ? '◀ left' : c.right ? 'right ▶' : 'straight', c.up ? 'gas' : c.down ? 'brake' : 'coast'];
    nodes[2].forEach(([nx, ny], i) => {
      g.fillStyle = '#aab0d4';
      g.fillText(OUTPUT_LABELS[i], nx + 9, ny - 2);
      g.fillStyle = '#ffd84a';
      g.fillText(decisions[i], nx + 9, ny + 10);
    });
  }

  let bannerTimer = null;
  function banner(html, seconds) {
    const el = $('banner');
    el.innerHTML = html;
    clearTimeout(bannerTimer);
    if (seconds) bannerTimer = setTimeout(() => { el.innerHTML = ''; }, seconds * 1000);
  }

  // ---------------------------------------------------------------------------
  // Controls
  // ---------------------------------------------------------------------------

  const stageSelect = $('stage');
  stageSelect.add(new Option('All styles (mixed)', String(MIXED)));
  STAGES.forEach((s, i) => stageSelect.add(new Option(`${i + 1} · ${s.name}`, String(i))));
  stageSelect.addEventListener('change', () => {
    if (lab.mode === 'watch') stopWatch();
    setStyle(Number(stageSelect.value));
    save();
    startGeneration(); // same population, new kind of road
  });

  const playBtn = $('play');
  function setRunning(on) {
    lab.running = on;
    playBtn.textContent = on ? '❚❚ Pause' : '▶ Train';
  }
  playBtn.addEventListener('click', () => setRunning(!lab.running));

  for (const b of document.querySelectorAll('[data-speed]')) {
    b.addEventListener('click', () => {
      lab.speed = b.dataset.speed === 'max' ? 'max' : Number(b.dataset.speed);
      for (const o of document.querySelectorAll('[data-speed]')) o.classList.toggle('selected', o === b);
      if (!lab.running) setRunning(true);
    });
  }
  for (const b of document.querySelectorAll('[data-camera]')) {
    b.addEventListener('click', () => {
      lab.cameraMode = b.dataset.camera;
      for (const o of document.querySelectorAll('[data-camera]')) o.classList.toggle('selected', o === b);
    });
  }

  $('watch').addEventListener('click', () => (lab.mode === 'watch' ? stopWatch() : startWatch()));
  // Two-click confirmation in the page itself: browser confirm() dialogs are
  // blocked in some embedded browsers, which made the button seem dead.
  const resetBtn = $('reset');
  let resetArmed = null;
  resetBtn.addEventListener('click', () => {
    if (resetArmed) {
      clearTimeout(resetArmed);
      resetArmed = null;
      resetBtn.textContent = 'Reset';
      resetBtn.classList.remove('armed');
      resetLab();
      banner('NEW START<small>40 brand-new random drivers</small>', 2);
      return;
    }
    resetBtn.textContent = 'Click again to reset';
    resetBtn.classList.add('armed');
    resetArmed = setTimeout(() => {
      resetArmed = null;
      resetBtn.textContent = 'Reset';
      resetBtn.classList.remove('armed');
    }, 4000);
  });

  document.addEventListener('visibilitychange', () => { if (document.hidden && lab.running) setRunning(false); });

  // ---------------------------------------------------------------------------
  // Main loop
  // ---------------------------------------------------------------------------

  const canvas = $('canvas');
  E.attach(canvas);
  const screenEl = $('screen');
  new ResizeObserver(() => {
    const r = screenEl.getBoundingClientRect();
    if (E.resizeView(r.width, r.height) && lab.theme) E.resetWeather(lab.theme);
    drawChart();
  }).observe(screenEl);

  let last = performance.now();
  let acc = 0;
  let statsTimer = 0;

  function frame(now) {
    const dt = Math.min(0.1, (now - last) / 1000);
    last = now;
    if (lab.running) {
      if (lab.mode === 'watch') {
        acc += dt;
        while (acc >= STEP && lab.mode === 'watch') { stepWatch(STEP); acc -= STEP; }
      } else if (lab.speed === 'max') {
        const until = performance.now() + 22;
        while (performance.now() < until) stepGeneration(STEP);
      } else {
        acc += dt * lab.speed;
        let steps = 0;
        while (acc >= STEP && steps++ < 4000) { stepGeneration(STEP); acc -= STEP; }
      }
    }
    renderRace(dt);
    updateHud();
    if ((statsTimer += dt) > 0.25) { statsTimer = 0; updateStats(); }
    requestAnimationFrame(frame);
  }

  if (!restore()) {
    setStyle(MIXED);
    lab.population = Array.from({ length: POPULATION }, randomGenome);
  }
  startGeneration();
  drawChart();
  requestAnimationFrame((t) => { last = t; frame(t); });

  // Test hook: ?debug exposes the lab and tools to check that learning is real.
  if (new URLSearchParams(location.search).has('debug')) {
    window.academy = {
      lab, INPUT_LABELS, TRAFFIC_INPUTS, GENOME, MIXED, randomGenome, simpleRule, testScore, solo,
      restart(styleIndex) {
        setStyle(styleIndex);
        resetLab();
      },
      train(generations) {
        const target = lab.generation + generations;
        const speed = lab.speed;
        lab.speed = 'max';
        while (lab.generation < target) stepGeneration(STEP);
        lab.speed = speed;
        return lab.history.slice(-generations);
      },
      // Full real stage with traffic, as "Watch champion" runs it, without drawing.
      realRace(genomeOrPolicy, stageIndex) {
        const track = realStage(stageIndex);
        const theme = THEMES[STAGES[stageIndex].theme];
        const policy = typeof genomeOrPolicy === 'function' ? genomeOrPolicy : null;
        const car = newCar(policy ? null : genomeOrPolicy);
        let t = 0;
        do { t += STEP; moveTraffic(track, STEP); } while (driveCar(car, track, theme, STEP, t, WATCH_LIMIT, policy));
        return { finished: car.finished, time: Math.round(t * 10) / 10, progress: Math.round((distance(car, track) / track.finish) * 100), bumps: car.crashes };
      },
      // Test-track score with some inputs zeroed (mask: one boolean per input).
      masked(genome, styleIndex, mask) {
        inputMask = mask;
        try { return testScore(genome, styleIndex); } finally { inputMask = null; }
      },
    };
  }
})();
