import Vue from 'vue'
import App from './App.vue'
import vuetify from './plugins/vuetify';
import store from "./store"
import Contextmenu from "vue-contextmenujs"
Vue.use(Contextmenu);
import VueLazyload from "vue-lazyload"
Vue.use(VueLazyload);
import vcolorpicker from 'vcolorpicker'
Vue.use(vcolorpicker);
// import animated from 'animate.css'
// Vue.use(animated)

Vue.config.productionTip = false;

Array.prototype.sum = function () { return this.reduce((sum, a) => sum + Number(a), 0); };
Array.prototype.mean = function () { return this.sum() / (this.length || 1); };
Array.prototype.max = function () { return Math.max(...this); };
Array.prototype.min = function () { return Math.min(...this); };
Array.prototype.mean2 = function () { return this[0].map((_, i) => this.map(arr => arr[i]).mean()); };
Array.prototype.shuffle = function(){ return this.map(a => [a,Math.random()]).sort((a,b) => {return a[1] < b[1] ? -1 : 1;}).map(a => a[0]); };
Array.prototype.sample = function(num){ return this.shuffle().slice(0,num); };
Array.prototype.quantile = function(q) {
  const sorted = this.slice().sort((a, b) => a - b);
  const pos = (sorted.length - 1) * q;
  const base = Math.floor(pos);
  const rest = pos - base;
  if (sorted[base + 1] !== undefined) {
      return sorted[base] + rest * (sorted[base + 1] - sorted[base]);
  } else {
      return sorted[base];
  }
}

Array.prototype.unique =
  Array.prototype.unique ||
  function () {
    return Array.from(new Set(this));
  };

new Vue({
  vuetify,
  store,
  render: h => h(App)
}).$mount('#app');

