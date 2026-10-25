// Animation timing is independent of display refresh rate and render cost.
export const SHOWCASE_DURATION = 45;
export const clamp01 = value => Math.max(0, Math.min(1, value));
export function easeMotion(progress) {
  const t = clamp01(progress);
  return t * t * t * (t * (6 * t - 15) + 10);
}

export class ShowcaseClock {
  constructor(duration = SHOWCASE_DURATION) {
    this.duration = duration;
    this.running = false;
    this.paused = false;
    this.elapsed = 0;
    this.last = 0;
    this.step = -1;
  }
  start(now) {
    this.running = true;
    this.paused = false;
    this.elapsed = 0;
    this.last = now;
    this.step = -1;
  }
  advance(now) {
    if (this.running && !this.paused) {
      this.elapsed = Math.min(this.duration, this.elapsed + Math.max(0, now - this.last) / 1000);
    }
    this.last = now;
    return this.elapsed;
  }
  setPaused(paused, now) {
    this.advance(now);
    this.paused = paused;
  }
  stop() {
    this.running = false;
    this.paused = false;
  }
  get finished() { return this.elapsed >= this.duration; }
  get progress() { return clamp01(this.elapsed / this.duration); }
}
