
export default class Highlight {
  constructor() {
    this.map = new Map();
    this.idx = new Set();
  }

  is_empty() { return this.map.size == 0; }


  add_highlight(source, highlight_idx) {
    this.map.set(source, new Set(highlight_idx));
    this.get_idx();
  }
  rm_highlight(source) {
    this.map.delete(source);
    this.get_idx();
  }
  clear() {
    this.map = new Map();
  }
  get_idx() {
    this.idx = new Set();
    for (let [key, value] of this.map) {
      value.forEach(v => this.idx.add(v));
    }
    return this.idx;
  }
  get_keys() {
    return Array.from(this.map.keys())
  }
  get_row_index() {
    const keys = this.get_keys().filter(d => d.includes("row"));
    let idx = new Set();
    for (let key of keys) {
      this.map.get(key).forEach(v => idx.add(v));
    }
    return Array.from(idx);
  }
  get_col_index() {
    const keys = this.get_keys().filter(d => d.includes("col"));
    let idx = new Set();
    for (let key of keys) {
      this.map.get(key).forEach(v => idx.add(v));
    }
    return Array.from(idx);
  }
  has(idx) {
    return this.is_empty() || this.idx.has(idx);
  }
  count(idx) {
    let cnt = 0;
    for (let elem of this.map) {
      if (elem[1].has(idx)) cnt += 1;
    }
    return cnt;
  }
}
