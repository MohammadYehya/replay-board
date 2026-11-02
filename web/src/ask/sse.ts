// Server-sent events read from a fetch() body (POST /ask streams them; EventSource cannot POST).
// Chunks can end anywhere, even inside a line or between "\r" and "\n": text is buffered until an event is whole.
// No DOM and no imports, so `npm test` can run it with plain Node.

export interface SseEvent {
  event: string; // "message" when the event names none
  data: string; // its data lines joined with "\n"
}

export class SseParser {
  private buf = '';
  private event = '';
  private data: string[] = [];
  private hasData = false;

  /** Feed decoded text; returns the events completed by it. */
  push(text: string): SseEvent[] {
    this.buf += text;
    const out: SseEvent[] = [];
    let start = 0;
    for (;;) {
      const nl = this.findLineEnd(start);
      if (nl < 0) break;
      const line = this.buf.slice(start, nl);
      start = nl + (this.buf[nl] === '\r' && this.buf[nl + 1] === '\n' ? 2 : 1);
      this.line(line, out);
    }
    this.buf = this.buf.slice(start);
    return out;
  }

  /** The stream ended: an event still open (no blank line after it) is dispatched. */
  end(): SseEvent[] {
    const out: SseEvent[] = [];
    if (this.buf) {
      this.line(this.buf.replace(/\r$/, ''), out);
      this.buf = '';
    }
    this.line('', out);
    return out;
  }

  /** Index of the next line end at or after `from`, or -1. A lone "\r" at the very end may be half of "\r\n". */
  private findLineEnd(from: number): number {
    for (let i = from; i < this.buf.length; i++) {
      const c = this.buf[i];
      if (c === '\n') return i;
      if (c === '\r') return i + 1 < this.buf.length ? i : -1;
    }
    return -1;
  }

  private line(line: string, out: SseEvent[]) {
    if (line === '') {
      if (this.hasData) out.push({ event: this.event || 'message', data: this.data.join('\n') });
      this.event = '';
      this.data = [];
      this.hasData = false;
      return;
    }
    if (line.startsWith(':')) return; // a comment (keep-alive)
    const colon = line.indexOf(':');
    const field = colon < 0 ? line : line.slice(0, colon);
    let value = colon < 0 ? '' : line.slice(colon + 1);
    if (value.startsWith(' ')) value = value.slice(1);
    if (field === 'event') this.event = value;
    else if (field === 'data') {
      this.data.push(value);
      this.hasData = true;
    }
  }
}
