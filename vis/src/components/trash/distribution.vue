<template>
  <g id='weight-distribution' :transform="`translate(${x},${y})`">
    <text :x="width / 2" :y="0" font-size=1.5rem style="text-anchor: middle; dominant-baseline: hanging;">Confidence</text>
    <g id='distribution-bars' :transform="`translate(${0},${legend_height + margin})`">
      <rect v-for="d in bars" :id='`distribution-bar-${d.label}`' :key='`distribution-bar-${d.label}`' :x="d.x" :y="d.y" :width="d.width" :height="d.height" :fill="d.color"></rect>
    </g>
    <g id='distribution-brush' :transform="`translate(${0},${bar_height + legend_height + 2 * margin})`">
      <rect id="distribution-overlay" :x="0" :y="0" :width="width" :height="handle_height" fill="transparent" stroke="black" stroke-width="1"></rect>
      <rect v-for="(d, i) in ranges" :key='`distribution-type-${i}`' :x="d.lower * width" :y="0" :width="(d.upper - d.lower) * width" :height="handle_height" stroke="black" stroke-width="1" :fill="d.color" @contextmenu.prevent="onContextmenu"></rect>
      <!-- <line v-for="(d, i) in separations" :id="`distribution-handle-${i}`" :key='`distribution-handle-${i}`' class="brush-handle" :x1="d.x * width" :y1="0" :x2="d.x * width" :y2="handle_height" :transform='`translate(${d.x},0)`' stroke="black" stroke-width="1" cursor="pointer"></line> -->
      <circle v-for="(d, i) in separations" :id="`distribution-handle-${i}`" :key='`distribution-handle-${i}`' class="brush-handle" :cx="d.x * width" :cy="handle_height / 2" :r="handle_height / 2 + margin / 2" :transform='`translate(${d.x},0)`' stroke="black" stroke-width="1" cursor="pointer" :fill='d.fixed ? "black" : "white" '></circle>
    </g>
  </g>
</template>

<script>
  import Vue from 'vue'
  import { mapState, mapMutations } from 'vuex'
  import { Shape, Color } from '@/plugins/utils.js';
  import * as d3 from 'd3';
  import { select, selectAll } from 'd3-selection';
  export default Vue.extend({
    name: 'Distribution',
    props: ['x','y','width','height'],
    mounted() {
      var drag = d3.drag()
          .on("start", () => {})
          .on("end", (event, d) => {
              d.prev_x = d.x;
              this.update_samples();
          })
          .on("drag", (event, d) => {
            if (d.fixed) return;
              var delta = event.x;
              d.x = d.prev_x + event.x / this.width;
              if (d.x > 1) d.x = 1;
              if (d.x < 0) d.x = 0;
              if (d.left_range_idx >= 0) {
                if (d.x <= this.ranges[d.left_range_idx].lower) d.x = this.ranges[d.left_range_idx].lower + 0.01;
              }
              if (d.right_range_idx >= 0) {
                if (d.x >= this.ranges[d.right_range_idx].upper) d.x = this.ranges[d.right_range_idx].upper - 0.01;
              }
              this.ranges[d.left_range_idx].upper = this.ranges[d.right_range_idx].lower = d.x;

              this.update_bars();
          });
      selectAll(".brush-handle")
          .data(this.separations)
          .call(drag);
    },
    data() {
      const AREA_COUNT = 4;
      const AREA_COLOR = [Color.RED_POINT, Color.WHITE_POINT, Color.GREEN_POINT, Color.DARK_GREEN];
      const DEFAULT_AREA_SEP = [0.0, 500/2140, 1000/2140, 1700/2140, 1.0];
      return {
        min_v: null,
        max_v: null,
        distribution: [],
        bars: [],
        separations: Array(AREA_COUNT + 1).fill(0).map((_, i) => Object({
          fixed: i === 0 || i === AREA_COUNT,
          left_range_idx: i === 0 ? -1 : i - 1,
          right_range_idx: i === AREA_COUNT ? -1 : i,
          x: DEFAULT_AREA_SEP[i],
          prev_x: DEFAULT_AREA_SEP[i]
        })),
        ranges: Array(AREA_COUNT).fill(0).map((_, i) => Object({
          lower: DEFAULT_AREA_SEP[i],
          upper: DEFAULT_AREA_SEP[i + 1],
          color: AREA_COLOR[i]
        })),
        bar_height: this.height - 50,
        handle_height: 5,
        margin: 10,
        legend_height: 25,
        area_count: AREA_COUNT,
        area_color: AREA_COLOR
      }
    },
    computed: {
      ...mapState(['all_samples', 'meta_data', 'color_map_list', 'redraw_cnt']),
      // bars() {
      //   const weights = Array.from(Array(this.meta_data.num_classes).keys()).map(l =>
      //       this.all_samples.filter(d => d && d.isVal && d.label==l).map(d => d.lam).sum());
      //   // const weights = Array.from(Array(this.meta_data.num_classes).keys()).map(l =>
      //   //     this.all_samples.filter(d => d && d.is_pos && d.label==l).map(d => d.weight).sum());
      //   const y_coef = this.height / weights.max() * 0.8;
      //   return weights.map((w,i) => ({
      //       'weight': w, 
      //       'label': i, 
      //       'color': this.color_map_list[i],
      //       'x': this.width / 14 * i,
      //       'y': this.height - y_coef * w,
      //       'height': y_coef * w,
      //       'width': this.width / 16}));
      // }
    },
    methods: {
      ...mapMutations(['redraw_all']),
      handleChangeColor() {

      },
      update_bars() {
          let distrib = this.distribution;
          let range_v = this.max_v - this.min_v;
          const y_coef = this.bar_height / distrib.map(w => w.length).max();
          var current_type = 0;
          this.bars = [];
          for (let i = 0; i < distrib.length; ++i) {
              let start = (distrib[i].x0 - this.min_v) / range_v,
                  end = (distrib[i].x1 - this.min_v) / range_v;
              let first_bar_idx = this.bars.length;
              while (this.ranges[current_type].upper < end) {
                  this.bars.push({
                      'label': start + "-" + this.ranges[current_type].upper, 
                      'y': this.bar_height - y_coef * distrib[i].length,
                      'height': y_coef * distrib[i].length,
                      'color': this.area_color[current_type],
                      'x': this.width * start,
                      'width': this.width * (this.ranges[current_type].upper - start)
                  });
                  start = this.ranges[current_type].upper;
                  current_type += 1;
              }
              this.bars.push({
                  'label': start + "-" + end, 
                  'y': this.bar_height - y_coef * distrib[i].length,
                  'height': y_coef * distrib[i].length,
                  'color': this.area_color[current_type],
                  'x': this.width * start,
                  'width': Math.max(0, this.width * (end - start) - 1)
              });
              this.bars[first_bar_idx].x += 1;
              this.bars[first_bar_idx].width = Math.max(0, this.bars[first_bar_idx].width - 1);
          }
      },
      update_samples() {
        let all_count = d3.sum(this.distribution.map(d => d.length)), count = 0;
        console.log(all_count);
        let threshold_idx = this.ranges.map(range => Math.round(range.upper * all_count) - 1);
        let threshold = []; // ( , ]
        for (let i = 0; i < this.distribution.length; ++i) {
            if (count + this.distribution[i].length <= threshold_idx[threshold.length]) {
                count += this.distribution[i].length;
            } else {
                let sorted = this.distribution[i].sort();
                while (count + this.distribution[i].length > threshold_idx[threshold.length]) {
                    threshold.push(sorted[threshold_idx[threshold.length] - count]);
                }
                count += this.distribution[i].length;
            } 
        }
        this.all_samples.forEach(sample => {
            if (!sample) return;
            let type = 0;
            while (type < this.area_count) {
                if (sample.meta_margin > threshold[type]) type += 1; else break;
            }
            sample.is_val = sample.is_pos = sample.is_neg = false;
            if (type == 0) {
              sample.is_neg = true;
            } else if (type == 2) {
              sample.is_pos = true;
            } else if (type == 3) {
              sample.is_val = true;
            }
            sample.color = this.area_color[type];
        });
        this.redraw_all();
      },
      onContextmenu(event) {
        const options = [['Change Color coding', null]];
        this.$contextmenu({
          items: options.map((option, l) => ({label:option[0], onClick:option[1]})),
          // items: options.map((option, l) => ({label:option[0], function(){this.subcluster()}})),
          event,
          customClass: "custom-class",
          zIndex: 3,
          minWidth: 130
        });
        return false;
      }
    },
    watch: {
      redraw_cnt: function() {
          const meta_margins = this.all_samples.map(d => d ? d.meta_margin : -1).filter(v => v > 0);
          this.distribution = d3.histogram().domain(d3.extent(meta_margins)).thresholds(20)(meta_margins);
          let distrib = this.distribution;
          this.min_v = distrib[0].x0 = distrib[0].x1 - (distrib[1].x1 - distrib[1].x0);
          this.max_v = distrib[distrib.length - 1].x1 = distrib[distrib.length - 1].x0 + (distrib[1].x1 - distrib[1].x0);
          // this.bars = distrib.map((w, i) => (
          //   return {
          //   'label': i, 
          //   'weight': w.length, 
          //   'y': this.bar_height - y_coef * w.length,
          //   'height': y_coef * w.length,
          //   'color': Color.lightgrey,
          //   'x': this.width / distrib.length * i,
          //   'width': this.width / distrib.length
          // }));
          this.update_bars();
          this.$forceUpdate();
      }
    }

});
</script>


<style>

</style>
