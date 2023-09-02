
const to_percent = (x) => (Math.round((x + Number.EPSILON) * 1000)/10).toString() + "%";
class Shape {
  static GET_STAR = (s) => `M0.0,${s*-3.2361},${s*0.7265},${s*-1.0},${s*3.0777},${s*-1.0},${s*1.1756},${s*0.382},${s*1.9021},${s*2.618},${s*0.0},${s*1.2361},${s*-1.9021},${s*2.618},${s*-1.1756},${s*0.382},${s*-3.0777},${s*-1.0},${s*-0.7265},${s*-1.0}Z`
  static GET_CIRCLE = (s) => `M${s},0A${s},${s},0,1,1,-${s},0A${s},${s},0,1,1,${s},0`
  static GET_DOWN_TRIANGLE = (s) => `M${s*-1/2},${s*-0.2886},${s*1/2},${s*-0.2886},${0},${s*0.2886*2}Z`;
  static GET_UP_TRIANGLE = (s) => `M${s*-1/2},${s*0.2886},${s*1/2},${s*0.2886},${0},${s*-0.2886*2}Z`;
  static GET_SQUARE = (s) => `M${-s},${-s},${-s},${s},${s},${s},${s},${-s}Z`;
  static STAR = this.GET_STAR(2.5);
  static CIRCLE = this.GET_CIRCLE(3);
  static UP_TRIANGLE = this.GET_UP_TRIANGLE(7);
  static DOWN_TRIANGLE = this.GET_DOWN_TRIANGLE(7);
  static SQUARE = this.GET_SQUARE(2.5)
  static LARGE_CIRCLE = this.GET_CIRCLE(5);
  static LARGE_UP_TRIANGLE = this.GET_UP_TRIANGLE(10);
  static LARGE_DOWN_TRIANGLE = this.GET_DOWN_TRIANGLE(10);
  static LARGE_SQUARE = this.GET_SQUARE(5);

  static GET_CROSS = (ratio) => `M ${ratio*5} ${ratio*3} L ${ratio*2} ${ratio*0} L ${ratio*5} ${ratio*-3} L ${ratio*3} ${ratio*-5} L ${ratio*0} ${ratio*-2} L ${ratio*-3} ${ratio*-5} L ${ratio*-5} ${ratio*-3} L ${ratio*-2} ${ratio*0} L ${ratio*-5} ${ratio*3} L ${ratio*-3} ${ratio*5} L ${ratio*0} ${ratio*2} L ${ratio*3} ${ratio*5} L ${ratio*5} ${ratio*3} Z`
  // static CROSS = this.GET_CROSS(0.9);
  static CROSS = this.GET_UP_TRIANGLE(8)

  static CLOSE_ICON = 'M 0 1 L 0 7 A 1 1 0 0 0 1 8 L 7 8 A 1 1 0 0 0 8 7 L 8 1 A 1 1 0 0 0 7 0 L 1 0 A 1 1 0 0 0 0 1 M 1 1 L 7 7 M 7 1 L 1 7';
  static EXPAND_ICON = 'M 2 1 L 2 7 L 5 4 L 2 1'
  static COLLAPSE_ICON = 'M 1 2 L 4 5 L 1 7 L 1 2'
  // static EXPAND_ICON = 'M 0 1 L 0 7 A 1 1 0 0 0 1 8 L 7 8 A 1 1 0 0 0 8 7 L 8 1 A 1 1 0 0 0 7 0 L 1 0 A 1 1 0 0 0 0 1 M 4 1 L 4 7 M 7 4 L 1 4';
  // static COLLAPSE_ICON = 'M 0 1 L 0 7 A 1 1 0 0 0 1 8 L 7 8 A 1 1 0 0 0 8 7 L 8 1 A 1 1 0 0 0 7 0 L 1 0 A 1 1 0 0 0 0 1 M 7 4 L 1 4';
}
class Color {
  static GRAY_BG = '#fafafa';
  static RED_BG = '#f5ded2';
  static GREEN_BG = '#e6f0e1';
  static WHITE_POINT = '#808080';
  static lightgrey = '#aaaaaa';
  static darkgrey = '#666666';
  static HIGHLIGHT = '#ff5500';
  static LIGHTGREEN = '#c9eadf';
  static LIGHTRED = '#f5dace';
  // static DARK_GREEN = '#35741d';
  // static GREEN_POINT = '#22b886';
  static DARK_GREEN = '#65af8f';
  static GREEN_POINT = '#65af8f';
  static RED_POINT = '#ca7772';
  static GRAY_POINT = '#aaaaaa';
}
const get_count = function(base, highlight) {
  return base.filter(i => highlight.has(i)).length;
}
const get_ratio = function(base, highlight) {
  return base.length == 0 ? 0 : get_count(base, highlight) / base.length;
}
const get_inner_height = (elem) =>  {
  const style = window.getComputedStyle(elem, null);
  return parseFloat(style.getPropertyValue("height")) - parseFloat(style.getPropertyValue("padding-top")) - parseFloat(style.getPropertyValue("padding-bottom"))
}
const get_inner_width = (elem) => {
  const style = window.getComputedStyle(elem, null);
  return parseFloat(style.getPropertyValue("width")) - parseFloat(style.getPropertyValue("padding-left")) - parseFloat(style.getPropertyValue("padding-right"))
}

function adjustRGBColorOpacity(r,g,b,a) {
  r = Math.floor(r * a + 255 * (1 - a));
  g = Math.floor(g * a + 255 * (1 - a));
  b = Math.floor(b * a + 255 * (1 - a));
  return rgb2color(r,g,b);
}

function adjustHexColorOpacity(col, a) {
  let [r,g,b] = hex2rgb(col);
  return adjustRGBColorOpacity(r,g,b, a);
}
function hex2rgb(col) {
  col = col.slice(1);
  const num = parseInt(col, 16);
  const r = num >> 16;
  const g = (num >> 8) & 0xff;
  const b = num & 0xff;
  return [r, g, b];
}
const hex = d => Number(d).toString(16).padStart(2, '0')
const rgb2color = (r, g, b) => "#" + hex(r)+hex(g)+hex(b);

const attr_shorten = (attr) => attr.replace("-epoch", "").replace("weight", "w").replace("margin", "m").replace("consistency", "c");

const check_top_bottom = (arr, key, k) => {
  for (let i = 0; i < k; ++i) if (arr[i]===key) return 1;
  for (let i = arr.length+k; i < arr.length; ++i) if (arr[i]===key) return 1;
  return 0;
}

const getTextWidth = (text, font) => {
  let canvas = getTextWidth.canvas || (getTextWidth.canvas = document.createElement("canvas"));
  let context = getTextWidth.context || (getTextWidth.context = canvas.getContext("2d"));
  context.font = font;
  return context.measureText(text).width;
}

const generate_polarity_function = (color1, color2) => {
  let [r1,g1,b1] = hex2rgb(color1);
  let [r2,g2,b2] = hex2rgb(color2);
  const thres = 0.25;
  return function(value, prior=0) {
    if (value > 0) return adjustRGBColorOpacity(r1,g1,b1, Math.min(thres + value/2*(1-thres), 1));
    if (value < 0) return adjustRGBColorOpacity(r2,g2,b2, Math.min(thres + -value/2*(1-thres), 1));
    return prior > 0 ? adjustRGBColorOpacity(r1,g1,b1,thres) : adjustRGBColorOpacity(r2,g2,b2,thres);
  }
}

const get_polarity_color = generate_polarity_function(Color.GREEN_POINT, Color.RED_POINT);


export {to_percent, Shape, Color, get_ratio, get_count, get_inner_height, get_inner_width, adjustHexColorOpacity, get_polarity_color, attr_shorten, check_top_bottom, getTextWidth};