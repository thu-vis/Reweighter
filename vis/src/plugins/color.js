
export default class ColorAssigner {
  constructor() {
    //this.color_pool = ['#1f78b4', '#ff7f00', '#6a3d9a', '#b15928', '#17becf', '#ffe895'];
    this.gray_color = '#aaaaaa';
    //this.color_pool = ['#a5c9e1','#ffcc99','#c3b1d6','#e0bca9','#a2e5ec','#ffcc99']
    //this.color_pool = ['#3366cc','#ff9900','#990099','#0099c6','#dd4477','#316395']
    //this.color_pool = this.color_pool.map(color => this.mix(color, '#ffffff', 0.6));
    this.color_pool = ['#91ace3','#995e14','#c266c2','#66c2dd','#2d6466',];
    this.color_pool = this.color_pool.map(color => this.mix(color, '#ffffff', 0.9));
    this.used = new Array(this.color_pool.length).fill(false);
  }

  mix(fg, bg, opacity) {
    function hex2rgb(col) {
      const num = parseInt(col.slice(1), 16);
      return [num >> 16, (num >> 8) & 0xff, num & 0xff];
    }
    const hex = d => Number(d).toString(16).padStart(2, '0')
    function rgb2hex(r, g, b) {
      return "#" + hex(r) + hex(g) + hex(b);
    }
    const [r1, g1, b1] = hex2rgb(fg);
    const [r2, g2, b2] = hex2rgb(bg);
    const [r, g, b] = [Math.round(opacity*r1+(1-opacity)*r2), Math.round(opacity*g1+(1-opacity)*g2), Math.round(opacity*b1+(1-opacity)*b2)];
    return rgb2hex(r, g, b);
  }

  init(label_names) {
    this.label_names = label_names;
    this.nclass = this.label_names.length;
    this.color_map = new Map();
    this.color_map_list = new Array(this.nclass).fill(this.gray_color);
    this.legend = [[-1,'Others',this.gray_color]];
    this.color_history = new Map();

    if (this.nclass < 6) {
      Array.from(Array(this.nclass).keys()).forEach(i => this.color_map_list[i] = this.color_pool[i])
      this.legend = Array.from(Array(this.nclass).keys()).map(i => [i,this.label_names[i],this.color_pool[i]]);
    }
  }

  update(labels) {
    if (this.nclass < 6) return;
    if (labels.length == 0) {
      this.color_map = new Map();
      this.color_map_list = new Array(this.nclass).fill(this.gray_color);
      this.legend = [[-1,'Others',this.gray_color]];
      this.color_history = new Map();
      return;
    }
    let counter = new Array(this.nclass).fill(0);
    labels.forEach(d => counter[d]++);
    counter = counter.map((cnt,i) => ({cnt, i}))
                     .filter(d => d.cnt > 3)
                     .sort((a,b) => (b.cnt-a.cnt))
                     .map(d => d.i)
                     .slice(0, this.color_pool.length);
    let previous_color_map = this.color_map;
    this.color_map = new Map();
    this.used = new Array(this.color_pool.length).fill(false);
    // fix color for case
    let FIX_DATA = [[2,'#91ace3'],[4,'#ffc266']];
    for (let [label, color] of FIX_DATA) {
      if (counter.includes(label)) {
        this.used[this.color_pool.indexOf(color)] = true;
        this.color_map.set(label, color);
        this.color_history.set(label, color);
      }
    }
    for (let [label, color] of previous_color_map) {
      if (counter.includes(label) && !this.used[this.color_pool.indexOf(color)]) {
        this.used[this.color_pool.indexOf(color)] = true;
        this.color_map.set(label, color);
        this.color_history.set(label, color);
      }
    }
    for (let [label, color] of this.color_history) {
      if (counter.includes(label) && !this.used[this.color_pool.indexOf(color)]) {
        this.used[this.color_pool.indexOf(color)] = true;
        this.color_map.set(label, color);
      }
    }
    for (let label of counter) {
      if (!this.color_map.has(label)) {
        let color_i = this.used.indexOf(false);
        this.used[color_i] = true;
        this.color_map.set(label, this.color_pool[color_i]);
      }
    }
    this.legend = [];
    this.color_map_list = new Array(this.nclass).fill(this.gray_color);
    for (let i = 0; i < this.nclass; ++i) {
      if (this.color_map.has(i))  {
        this.legend.push([i, this.label_names[i], this.color_map.get(i)]);
        this.color_map_list[i] = this.color_map.get(i);
      }
    }
    this.legend.push([-1,'Others',this.gray_color]);
  }
}
