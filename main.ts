// roni — sokoban-solver-01: room-wise iterative-deepening A* (IDA*) over pushes.
//
// The generator lays a hallway across the board with one door per room, and a box
// pushed through a door can never come back, so every box stays in its room and a
// level's push-optimal count is the sum of its rooms' optima. This solver splits
// the level into its rooms, solves each room on its own from its door, and walks
// the player between doors through the hallway. Inside a room the search is IDA*
// over push states (the player is normalised to the region it can reach), with:
//   - an admissible heuristic: the cheapest assignment of boxes to goals over
//     shortest-path distances on the room's floor (exact min-cost matching, n<=4),
//   - a deadlock table built once per room by reverse pull-reachability from the
//     goals (a square no pull sequence can reach from a goal can never hold a box
//     that ends on a goal), plus a 2x2 freeze check at every generated state,
//   - a transposition table keyed by (sorted boxes, player-region) holding the
//     cheapest pushes seen, which prunes re-expansions without losing optimality.
// A push-based A* with the same heuristic re-checks any room whose IDA* answer
// sits above the heuristic lower bound, and the cheaper of the two wins, so a
// room answer is push-optimal whenever either search completes. If a room cannot
// be split or solved this way, the whole level falls back to the same push A*.
// Work is bounded by node-visit caps, never by the wall clock, and nothing here
// reads the clock, randomness or the environment, so two runs are byte-identical.
import { mkdirSync, readdirSync, readFileSync, writeFileSync } from 'node:fs';
import { join } from 'node:path';

const I = process.env.MAKEFUN_INPUT_DIR ?? '/input';
const O = process.env.MAKEFUN_OUTPUT_DIR ?? '/output';

// Work budgets (node visits / expansions), never the wall clock.
const IDA_VISIT_CAP = 400_000;
const ASTAR_EXPAND_CAP = 250_000;
const WHOLE_EXPAND_CAP = 400_000;

const DIRS: ReadonlyArray<readonly [number, number, string]> = [
  [0, -1, 'u'],
  [0, 1, 'd'],
  [-1, 0, 'l'],
  [1, 0, 'r'],
];

interface Level {
  w: number;
  h: number;
  wall: boolean[];
  goals: number[];
  goalSet: Set<number>;
  boxes: number[];
  player: number;
}

function parseLevel(text: string): Level {
  const rows = text.split('\n').filter((r) => r.length > 0);
  const h = rows.length;
  const w = Math.max(...rows.map((r) => r.length));
  const wall: boolean[] = new Array(w * h).fill(true);
  const goals: number[] = [];
  const goalSet = new Set<number>();
  const boxes: number[] = [];
  let player = 0;
  rows.forEach((row, y) => {
    for (let x = 0; x < w; x++) {
      const ch = row[x] ?? '#';
      const i = y * w + x;
      wall[i] = ch === '#';
      if (ch === '.' || ch === '*' || ch === '+') {
        goals.push(i);
        goalSet.add(i);
      }
      if (ch === '$' || ch === '*') boxes.push(i);
      if (ch === '@' || ch === '+') player = i;
    }
  });
  boxes.sort((a, b) => a - b);
  return { w, h, wall, goals, goalSet, boxes, player };
}

// Shortest-path distances from one source over the cells marked allowed.
function bfsDist(src: number, allowed: boolean[], w: number, h: number): number[] {
  const dist: number[] = new Array(w * h).fill(Infinity);
  dist[src] = 0;
  const q: number[] = [src];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!allowed[n] || dist[n] !== Infinity) continue;
      dist[n] = dist[c]! + 1;
      q.push(n);
    }
  }
  return dist;
}

// The player's walk from `from` to `to` avoiding `blocked`, as LURD letters.
function bfsWalk(
  from: number,
  to: number,
  blocked: Set<number>,
  allowed: boolean[],
  w: number,
  h: number,
): string | null {
  if (from === to) return '';
  const prev = new Map<number, [number, string]>();
  const q: number[] = [from];
  prev.set(from, [-1, '']);
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const [dx, dy, letter] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!allowed[n] || blocked.has(n) || prev.has(n)) continue;
      prev.set(n, [c, letter]);
      if (n === to) {
        let s = '';
        let cur = n;
        while (cur !== from) {
          const [p, l] = prev.get(cur)!;
          s = l + s;
          cur = p;
        }
        return s;
      }
      q.push(n);
    }
  }
  return null;
}

// Every cell the player can reach from `start` without stepping on `blocked`.
function floodRegion(
  start: number,
  blocked: Set<number>,
  allowed: boolean[],
  w: number,
  h: number,
): Set<number> {
  const seen = new Set<number>([start]);
  const q: number[] = [start];
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const [dx, dy] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!allowed[n] || blocked.has(n) || seen.has(n)) continue;
      seen.add(n);
      q.push(n);
    }
  }
  return seen;
}

// Deadlock table: a box square is dead when no sequence of pulls can bring a box
// from a goal onto it (other boxes ignored, so this only ever over-approximates
// liveness and never prunes a winnable state). Built by searching backwards:
// from (goal, adjacent player) states, the player walks anywhere beside the box
// and pulls it one square towards itself.
function computeDead(
  boxAllowed: boolean[],
  playerAllowed: boolean[],
  goals: number[],
  w: number,
  h: number,
): Set<number> {
  const n = w * h;
  const live = new Set<number>();
  const seen = new Set<number>();
  const qb: number[] = [];
  const qp: number[] = [];
  const add = (b: number, p: number): void => {
    const k = b * n + p;
    if (seen.has(k)) return;
    seen.add(k);
    live.add(b);
    qb.push(b);
    qp.push(p);
  };
  for (const g of goals) {
    const gx = g % w;
    const gy = Math.floor(g / w);
    for (const [dx, dy] of DIRS) {
      const nx = gx + dx;
      const ny = gy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const p = ny * w + nx;
      if (playerAllowed[p]) add(g, p);
    }
  }
  for (let qi = 0; qi < qb.length; qi++) {
    const b = qb[qi]!;
    const p = qp[qi]!;
    const region = floodRegion(p, new Set<number>([b]), playerAllowed, w, h);
    const bx = b % w;
    const by = Math.floor(b / w);
    for (const [dx, dy] of DIRS) {
      const nx = bx + dx;
      const ny = by + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const q = ny * w + nx;
      if (!region.has(q)) continue; // the player must be able to stand here
      if (!boxAllowed[q]) continue; // and the box must be able to move here
      // Pull: the player at q drags the box from b onto q, stepping back to
      // the square beyond q (2q - b). That square is where the player ends.
      const rx = nx + dx;
      const ry = ny + dy;
      if (rx < 0 || ry < 0 || rx >= w || ry >= h) continue;
      const r = ry * w + rx;
      if (!playerAllowed[r]) continue;
      add(q, r);
    }
  }
  const dead = new Set<number>();
  for (let c = 0; c < n; c++) if (boxAllowed[c] && !live.has(c)) dead.add(c);
  return dead;
}

// 2x2 blocks whose cells are all wall or box-occupiable: a freeze candidate.
function computeBlocks(boxAllowed: boolean[], wall: boolean[], w: number, h: number): number[][] {
  const blocks: number[][] = [];
  for (let y = 0; y + 1 < h; y++) {
    for (let x = 0; x + 1 < w; x++) {
      const cells = [y * w + x, y * w + x + 1, (y + 1) * w + x, (y + 1) * w + x + 1];
      if (cells.every((c) => wall[c] || boxAllowed[c])) blocks.push(cells);
    }
  }
  return blocks;
}

interface Ctx {
  w: number;
  h: number;
  wall: boolean[];
  boxAllowed: boolean[]; // cells a box may occupy
  playerAllowed: boolean[]; // cells the player may stand on
  goals: number[];
  goalSet: Set<number>;
  distToGoal: number[][]; // per goal, BFS distance over boxAllowed
  dead: Set<number>;
  blocks: number[][];
  useAssignment: boolean; // exact min-cost matching only while n <= 4
}

// Exact cheapest box-to-goal assignment over shortest-path distances (n <= 4,
// so every permutation is tried). Admissible: one push moves one box one square.
function assignmentCost(boxes: number[], distToGoal: number[][]): number {
  const n = boxes.length;
  const idx = boxes.map((_, i) => i);
  let best = Infinity;
  const permute = (k: number, acc: number): void => {
    if (acc >= best) return;
    if (k === n) {
      best = acc;
      return;
    }
    for (let j = k; j < n; j++) {
      const t = idx[k]!;
      idx[k] = idx[j]!;
      idx[j] = t;
      const d = distToGoal[idx[k]!]![boxes[k]!]!;
      if (d !== Infinity) permute(k + 1, acc + d);
      const t2 = idx[k]!;
      idx[k] = idx[j]!;
      idx[j] = t2;
    }
  };
  permute(0, 0);
  return best;
}

// Sum of each box's nearest-goal distance: admissible, used when n > 4.
function minSumCost(boxes: number[], distToGoal: number[][]): number {
  let s = 0;
  for (const b of boxes) {
    let m = Infinity;
    for (let j = 0; j < distToGoal.length; j++) m = Math.min(m, distToGoal[j]![b]!);
    if (m === Infinity) return Infinity;
    s += m;
  }
  return s;
}

function heuristic(ctx: Ctx, boxes: number[]): number {
  return ctx.useAssignment ? assignmentCost(boxes, ctx.distToGoal) : minSumCost(boxes, ctx.distToGoal);
}

function frozen(ctx: Ctx, boxes: number[]): boolean {
  const boxSet = new Set(boxes);
  for (const blk of ctx.blocks) {
    let filled = true;
    let bad = false;
    for (const c of blk) {
      if (boxSet.has(c)) {
        if (!ctx.goalSet.has(c)) bad = true;
      } else if (!ctx.wall[c]) {
        filled = false;
        break;
      }
    }
    if (filled && bad) return true;
  }
  return false;
}

interface PushChild {
  boxes: number[]; // sorted
  player: number; // cell the player ends on (the box's old square)
  walk: string; // player steps from the parent's square to the pushing stance
  push: string; // the push letter
  h: number;
}

// One walk-BFS from `from` over the reachable floor, as parent pointers, so
// every push in one expansion reads its walk off the same map.
function walkParents(
  from: number,
  blocked: Set<number>,
  allowed: boolean[],
  w: number,
  h: number,
): Map<number, [number, string]> {
  const prev = new Map<number, [number, string]>();
  const q: number[] = [from];
  prev.set(from, [-1, '']);
  for (let qi = 0; qi < q.length; qi++) {
    const c = q[qi]!;
    const cx = c % w;
    const cy = Math.floor(c / w);
    for (const [dx, dy, letter] of DIRS) {
      const nx = cx + dx;
      const ny = cy + dy;
      if (nx < 0 || ny < 0 || nx >= w || ny >= h) continue;
      const n = ny * w + nx;
      if (!allowed[n] || blocked.has(n) || prev.has(n)) continue;
      prev.set(n, [c, letter]);
      q.push(n);
    }
  }
  return prev;
}

function walkFrom(parents: Map<number, [number, string]>, from: number, to: number): string {
  let s = '';
  let cur = to;
  while (cur !== from) {
    const [p, l] = parents.get(cur)!;
    s = l + s;
    cur = p;
  }
  return s;
}

// Every legal push from a state: each box, each direction, where the player can
// reach the pushing stance. Dead squares and 2x2 freezes are cut at birth.
function genPushes(ctx: Ctx, boxes: number[], player: number): PushChild[] {
  const boxSet = new Set(boxes);
  const parents = walkParents(player, boxSet, ctx.playerAllowed, ctx.w, ctx.h);
  const out: PushChild[] = [];
  for (let bi = 0; bi < boxes.length; bi++) {
    const b = boxes[bi]!;
    const bx = b % ctx.w;
    const by = Math.floor(b / ctx.w);
    for (let di = 0; di < 4; di++) {
      const [dx, dy, letter] = DIRS[di]!;
      const nx = bx + dx;
      const ny = by + dy;
      if (nx < 0 || ny < 0 || nx >= ctx.w || ny >= ctx.h) continue;
      const dest = ny * ctx.w + nx;
      if (!ctx.boxAllowed[dest] || boxSet.has(dest)) continue;
      const sx = bx - dx;
      const sy = by - dy;
      if (sx < 0 || sy < 0 || sx >= ctx.w || sy >= ctx.h) continue;
      const stance = sy * ctx.w + sx;
      if (!parents.has(stance)) continue;
      const nb = boxes.slice();
      nb[bi] = dest;
      nb.sort((a, c) => a - c);
      if (nb.some((c) => ctx.dead.has(c))) continue;
      if (frozen(ctx, nb)) continue;
      const h = heuristic(ctx, nb);
      if (h === Infinity) continue;
      out.push({ boxes: nb, player: b, walk: walkFrom(parents, player, stance), push: letter, h });
    }
  }
  return out;
}

interface SolveResult {
  moves: string;
  pushes: number;
}

interface PushNode {
  boxes: number[];
  player: number;
  g: number;
  h: number;
  parent: number;
  walk: string;
  push: string;
}

function rebuild(nodes: PushNode[], found: number): SolveResult {
  const parts: string[] = [];
  let cur = found;
  while (cur > 0) {
    const nd = nodes[cur]!;
    parts.push(nd.walk + nd.push);
    cur = nd.parent;
  }
  parts.reverse();
  return { moves: parts.join(''), pushes: nodes[found]!.g };
}

const keyOf = (boxes: number[], player: number): string => boxes.join(',') + '|' + player;

// Iterative-deepening A* over pushes. The f-limit starts at the heuristic lower
// bound and rises to the smallest f cut off, so the first solution found at a
// limit is push-optimal; the transposition table only prunes a state another
// path already reached with no more pushes, which cannot hide the optimum.
function idaStar(ctx: Ctx, startBoxes: number[], startPlayer: number): SolveResult | null {
  const h0 = heuristic(ctx, startBoxes);
  if (h0 === Infinity) return null;
  if (h0 === 0) return { moves: '', pushes: 0 };
  const nodes: PushNode[] = [
    {
      boxes: startBoxes,
      player: startPlayer,
      g: 0,
      h: h0,
      parent: -1,
      walk: '',
      push: '',
    },
  ];
  const tt = new Map<string, number>();
  let visits = 0;
  let overCap = false;
  let foundIdx = -1;
  let minNext = Infinity;

  const dfs = (idx: number, limit: number): boolean => {
    if (overCap) return false;
    if (++visits > IDA_VISIT_CAP) {
      overCap = true;
      return false;
    }
    const node = nodes[idx]!;
    const k = keyOf(node.boxes, node.player);
    const seen = tt.get(k);
    if (seen !== undefined && seen <= node.g) return false;
    tt.set(k, node.g);
    if (node.h === 0) {
      foundIdx = idx;
      return true;
    }
    const kids = genPushes(ctx, node.boxes, node.player);
    kids.sort(
      (a, b) =>
        node.g + 1 + a.h - (node.g + 1 + b.h) ||
        a.h - b.h ||
        (a.boxes.join(',') < b.boxes.join(',') ? -1 : 1) ||
        (a.push < b.push ? -1 : 1),
    );
    for (const ch of kids) {
      const f = node.g + 1 + ch.h;
      if (f > limit) {
        if (f < minNext) minNext = f;
        continue;
      }
      const cIdx = nodes.length;
      nodes.push({
        boxes: ch.boxes,
        player: ch.player,
        g: node.g + 1,
        h: ch.h,
        parent: idx,
        walk: ch.walk,
        push: ch.push,
      });
      if (dfs(cIdx, limit)) return true;
      if (overCap) return false;
    }
    return false;
  };

  let limit = h0;
  for (;;) {
    minNext = Infinity;
    foundIdx = -1;
    overCap = false;
    tt.clear(); // fresh table each round: an entry from a lower limit would
    // otherwise prune the root (and any state) on re-entry at this limit
    if (dfs(0, limit)) return rebuild(nodes, foundIdx);
    if (overCap) return null;
    if (minNext === Infinity) return null;
    limit = minNext;
  }
}

// Push-based A* with the same heuristic and transposition table. Used to
// re-check a room whose IDA* answer sits above the lower bound, and as the
// fallback when IDA* runs out of its visit budget.
function aStar(ctx: Ctx, startBoxes: number[], startPlayer: number, cap: number): SolveResult | null {
  const h0 = heuristic(ctx, startBoxes);
  if (h0 === Infinity) return null;
  if (h0 === 0) return { moves: '', pushes: 0 };
  const nodes: PushNode[] = [
    {
      boxes: startBoxes,
      player: startPlayer,
      g: 0,
      h: h0,
      parent: -1,
      walk: '',
      push: '',
    },
  ];
  const tt = new Map<string, number>([[keyOf(startBoxes, startPlayer), 0]]);
  // Binary heap on [f, h, insertion order, node index].
  const heap: [number, number, number, number][] = [[h0, h0, 0, 0]];
  let seq = 1;
  let expansions = 0;
  const less = (a: [number, number, number, number], b: [number, number, number, number]): boolean =>
    a[0] < b[0] || (a[0] === b[0] && (a[1] < b[1] || (a[1] === b[1] && a[2] < b[2])));
  const heapPush = (e: [number, number, number, number]): void => {
    let i = heap.length;
    heap.push(e);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (!less(heap[i]!, heap[p]!)) break;
      const t = heap[i]!;
      heap[i] = heap[p]!;
      heap[p] = t;
      i = p;
    }
  };
  const heapPop = (): [number, number, number, number] => {
    const top = heap[0]!;
    const last = heap.pop()!;
    if (heap.length > 0) {
      heap[0] = last;
      for (let i = 0; ; ) {
        let c = 2 * i + 1;
        if (c >= heap.length) break;
        if (c + 1 < heap.length && less(heap[c + 1]!, heap[c]!)) c++;
        if (!less(heap[c]!, heap[i]!)) break;
        const t = heap[i]!;
        heap[i] = heap[c]!;
        heap[c] = t;
        i = c;
      }
    }
    return top;
  };

  while (heap.length > 0) {
    const [, , , idx] = heapPop();
    const node = nodes[idx]!;
    const k = keyOf(node.boxes, node.player);
    if (tt.get(k) !== node.g) continue; // a cheaper path to this state was queued
    if (node.h === 0) return rebuild(nodes, idx);
    if (++expansions > cap) return null;
    for (const ch of genPushes(ctx, node.boxes, node.player)) {
      const g2 = node.g + 1;
      const k2 = keyOf(ch.boxes, ch.player);
      const seen = tt.get(k2);
      if (seen !== undefined && seen <= g2) continue;
      tt.set(k2, g2);
      const cIdx = nodes.length;
      nodes.push({
        boxes: ch.boxes,
        player: ch.player,
        g: g2,
        h: ch.h,
        parent: idx,
        walk: ch.walk,
        push: ch.push,
      });
      heapPush([g2 + ch.h, ch.h, seq++, cIdx]);
    }
  }
  return null;
}

// IDA* first; when its answer is above the heuristic lower bound (so optimality
// is not already proven), A* re-checks and the cheaper answer wins. When IDA*
// runs out of budget, A* is the fallback.
function solveWithCtx(ctx: Ctx, boxes: number[], player: number): SolveResult | null {
  if (boxes.some((b) => ctx.dead.has(b))) return null;
  const h0 = heuristic(ctx, boxes);
  const ida = idaStar(ctx, boxes, player);
  if (ida === null) return aStar(ctx, boxes, player, ASTAR_EXPAND_CAP);
  if (ida.pushes > h0) {
    const verify = aStar(ctx, boxes, player, ASTAR_EXPAND_CAP);
    if (verify !== null && verify.pushes < ida.pushes) return verify;
  }
  return ida;
}

function roomCtx(level: Level, cells: Set<number>, entry: number): Ctx {
  const n = level.w * level.h;
  const boxAllowed: boolean[] = new Array(n).fill(false);
  const playerAllowed: boolean[] = new Array(n).fill(false);
  for (const c of cells) {
    boxAllowed[c] = true;
    playerAllowed[c] = true;
  }
  playerAllowed[entry] = true; // the player enters through the door
  const goals = level.goals.filter((g) => cells.has(g));
  const distToGoal = goals.map((g) => bfsDist(g, boxAllowed, level.w, level.h));
  return {
    w: level.w,
    h: level.h,
    wall: level.wall,
    boxAllowed,
    playerAllowed,
    goals,
    goalSet: new Set(goals),
    distToGoal,
    dead: computeDead(boxAllowed, playerAllowed, goals, level.w, level.h),
    blocks: computeBlocks(boxAllowed, level.wall, level.w, level.h),
    useAssignment: goals.length <= 4,
  };
}

// Apply a move string to a state; null when a step is illegal.
function applyMoves(
  level: Level,
  player0: number,
  boxes0: number[],
  moves: string,
): { player: number; boxes: number[] } | null {
  let player = player0;
  const boxes = boxes0.slice();
  const boxAt = new Map<number, number>();
  boxes.forEach((b, i) => boxAt.set(b, i));
  for (const ch of moves) {
    const d = { u: [0, -1], d: [0, 1], l: [-1, 0], r: [1, 0] }[ch.toLowerCase() as 'u' | 'd' | 'l' | 'r'];
    if (!d) return null;
    const nx = (player % level.w) + d[0];
    const ny = Math.floor(player / level.w) + d[1];
    if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) return null;
    const dest = ny * level.w + nx;
    if (level.wall[dest]) return null;
    const bi = boxAt.get(dest);
    if (bi !== undefined) {
      const bx = nx + d[0];
      const by = ny + d[1];
      if (bx < 0 || by < 0 || bx >= level.w || by >= level.h) return null;
      const beyond = by * level.w + bx;
      if (level.wall[beyond] || boxAt.has(beyond)) return null;
      boxAt.delete(dest);
      boxAt.set(beyond, bi);
      boxes[bi] = beyond;
    }
    player = dest;
  }
  return { player, boxes };
}

interface Room {
  cells: Set<number>;
  boxes: number[];
  goals: number[];
  entries: number[];
}

// Split the level into rooms: connected floor off the hallway row (the row the
// player starts in). Each room holds its boxes and goals and one door onto the
// hallway. Null when the split does not cover every box cleanly.
function tryDecomposed(level: Level): string | null {
  const hallwayRow = Math.floor(level.player / level.w);
  const seen = new Set<number>();
  const rooms: Room[] = [];
  for (let c = 0; c < level.w * level.h; c++) {
    if (level.wall[c] || Math.floor(c / level.w) === hallwayRow || seen.has(c)) continue;
    const cells = new Set<number>([c]);
    const q: number[] = [c];
    seen.add(c);
    for (let qi = 0; qi < q.length; qi++) {
      const cur = q[qi]!;
      const cx = cur % level.w;
      const cy = Math.floor(cur / level.w);
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) continue;
        const n = ny * level.w + nx;
        if (level.wall[n] || Math.floor(n / level.w) === hallwayRow || seen.has(n)) continue;
        seen.add(n);
        cells.add(n);
        q.push(n);
      }
    }
    const entries = new Set<number>();
    for (const cell of cells) {
      const cx = cell % level.w;
      const cy = Math.floor(cell / level.w);
      for (const [dx, dy] of DIRS) {
        const nx = cx + dx;
        const ny = cy + dy;
        if (nx < 0 || ny < 0 || nx >= level.w || ny >= level.h) continue;
        const n = ny * level.w + nx;
        if (Math.floor(n / level.w) === hallwayRow && !level.wall[n]) entries.add(n);
      }
    }
    rooms.push({
      cells,
      boxes: level.boxes.filter((b) => cells.has(b)),
      goals: level.goals.filter((g) => cells.has(g)),
      entries: [...entries].sort((a, b) => a - b),
    });
  }
  const boxed = new Set<number>();
  for (const r of rooms) for (const b of r.boxes) boxed.add(b);
  if (boxed.size !== level.boxes.length) return null;
  for (const r of rooms) {
    if (r.boxes.length === 0) continue;
    if (r.boxes.length !== r.goals.length || r.entries.length === 0) return null;
  }
  const active = rooms
    .filter((r) => r.boxes.length > 0)
    .sort((a, b) => a.entries[0]! - b.entries[0]!);
  const allFloor = level.wall.map((w2) => !w2);
  let moves = '';
  let player = level.player;
  const curBoxes = new Set<number>(level.boxes);
  for (const room of active) {
    const entry = room.entries[0]!;
    const walk = bfsWalk(player, entry, curBoxes, allFloor, level.w, level.h);
    if (walk === null) return null;
    moves += walk;
    const ctx = roomCtx(level, room.cells, entry);
    const res = solveWithCtx(ctx, room.boxes.slice().sort((a, b) => a - b), entry);
    if (res === null) return null;
    const fin = applyMoves(level, entry, room.boxes, res.moves);
    if (fin === null) return null;
    moves += res.moves;
    for (const b of room.boxes) curBoxes.delete(b);
    for (const b of fin.boxes) curBoxes.add(b);
    player = fin.player;
  }
  if (![...curBoxes].every((b) => level.goalSet.has(b))) return null;
  const check = applyMoves(level, level.player, level.boxes, moves);
  if (check === null || !check.boxes.every((b) => level.goalSet.has(b))) return null;
  return moves;
}

// Whole-level fallback: the same push A* over the entire board, used when the
// room split does not apply or a room cannot be solved on its own.
function solveWhole(level: Level): string | null {
  const n = level.w * level.h;
  const boxAllowed = level.wall.map((w2) => !w2);
  const playerAllowed = boxAllowed.slice();
  const distToGoal = level.goals.map((g) => bfsDist(g, boxAllowed, level.w, level.h));
  const ctx: Ctx = {
    w: level.w,
    h: level.h,
    wall: level.wall,
    boxAllowed,
    playerAllowed,
    goals: level.goals,
    goalSet: level.goalSet,
    distToGoal,
    dead: computeDead(boxAllowed, playerAllowed, level.goals, level.w, level.h),
    blocks: computeBlocks(boxAllowed, level.wall, level.w, level.h),
    useAssignment: level.goals.length <= 4,
  };
  const res = aStar(ctx, level.boxes, level.player, WHOLE_EXPAND_CAP);
  if (res === null) return null;
  const fin = applyMoves(level, level.player, level.boxes, res.moves);
  if (fin === null || !fin.boxes.every((b) => level.goalSet.has(b))) return null;
  return res.moves;
}

function solveLevel(text: string): string {
  const level = parseLevel(text);
  return tryDecomposed(level) ?? solveWhole(level) ?? '';
}

for (const id of readdirSync(I).sort()) {
  const moves = solveLevel(readFileSync(join(I, id, 'level.txt'), 'utf8'));
  mkdirSync(join(O, id), { recursive: true });
  writeFileSync(join(O, id, 'solution.txt'), `${moves}\n`);
}
