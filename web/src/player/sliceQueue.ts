// Work that must not freeze the page in one go, done in short slices between frames. The board's typeset layer fits
// each reading to its box with a few forced layouts; done all at once when the player opened, an hour-long lecture
// (252 pieces, 58 lines, 10k transcript words in the page) froze it for 3 s twice (measured: the player and its
// transcript showed after 3.5 s, and 8 s of long tasks in the first 10 s).

export type Scheduler = (run: () => void) => void;

export class SliceQueue<K = unknown> {
  private readonly jobs = new Map<K, () => void>();
  private readonly budgetMs: number;
  private readonly now: () => number;
  private readonly schedule: Scheduler;
  private scheduled = false;

  /** budgetMs: how long one slice may run (it always runs at least one job). */
  constructor(budgetMs = 12, now: () => number = () => performance.now(), schedule: Scheduler = (run) => setTimeout(run, 0)) {
    this.budgetMs = budgetMs;
    this.now = now;
    this.schedule = schedule;
  }

  /** Queue a job. A job still waiting under the same key is replaced (only the latest one matters). */
  add(key: K, job: () => void) {
    this.jobs.delete(key);
    this.jobs.set(key, job);
    this.kick();
  }

  /** Drop the job waiting under this key, if any. */
  cancel(key: K) {
    this.jobs.delete(key);
  }

  get size(): number {
    return this.jobs.size;
  }

  /** One slice: jobs in the order queued until the budget is spent; the rest wait for the next slice. A job that throws
   *  is dropped and the others still run. */
  run() {
    this.scheduled = false;
    const t0 = this.now();
    for (const [key, job] of this.jobs) {
      this.jobs.delete(key);
      try {
        job();
      } catch {
        // one reading that cannot be fitted keeps its handwriting; the others still get fitted
      }
      if (this.now() - t0 >= this.budgetMs) break;
    }
    this.kick();
  }

  private kick() {
    if (this.scheduled || !this.jobs.size) return;
    this.scheduled = true;
    this.schedule(() => this.run());
  }
}
