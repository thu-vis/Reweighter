<template>
    <v-col cols='12' class='fill-width' id='scatterplot-column'>
    <v-col cols='12' class='topname fill-width'>Scatterplot</v-col>
    <svg v-if='redrawToggle === true' :width='svgWidth' :height='svgHeight' :viewBox='`${-margin.left},${-margin.top},${viewWidth+margin.left+margin.right},${viewHeight+margin.top+margin.bottom}`'>
      <g id='point-canvas'><rect :x='-margin.left' :y='-margin.top' :width='viewWidth+margin.left+margin.right' :height='viewWidth+margin.top+margin.bottom' fill=#ffffff></rect>
       <path v-for="d in samples" :key="`graph-scatter-${d.idx}`" class="scatter" :transform="`translate(${d.coordx},${d.coordy})`"
          :d="highlight_idx.has(d.idx) ? d.large_shape : d.shape" :fill="d.color" :fill-opacity="highlight_idx.has(d.idx) ? 1 : .5"/>
      </g> 
      <g id='line-canvas' :transform='`translate(${-margin.left},${viewWidth+margin.top})`'></g>
    </svg>
    </v-col>
</template>

<script>
  import Vue from 'vue'
  
  import { mapState, mapMutations } from 'vuex'
  import { scaleLinear } from 'd3-scale';
  import { extent } from 'd3-array';
  import { select, selectAll } from 'd3-selection';
  import { Shape, get_inner_width } from '@/plugins/utils.js';
  import { lasso } from 'd3-lasso';
  import { line } from 'd3-shape';
  import { brushX } from 'd3-brush';

  export default Vue.extend({
    name: 'Graph',
    mounted() {
      this.resize();
      this.update_view();
      window.addEventListener('resize', () => {
        this.redrawToggle = false;
        setTimeout(() => {
          this.redrawToggle = true;
          this.resize();
          this.update_view();
        }, 300);
      });
    },
    data() {
      return {
        svgWidth: 0,
        redrawToggle: true,
        viewWidth: 500,
        viewHeight: 500,
        margin: {'left': 5, 'right': 5, 'top': 5, 'bottom': 5},
        brush: null,
        ENABLE_OPACITY: true,
        current_selection: [0, 10],
        fast_average: null,
      }
    },
    methods: {
      ...mapMutations(['add_highlight_idx', 'rm_highlight_idx', 'set_color_opacity']),
      resize() {
        this.svgWidth = get_inner_width(document.getElementById('scatterplot-column'));
      },
      update_view() {
        this.samples.forEach(d => {
          d.coordx = this.xScale(d.x);
          d.coordy = this.yScale(d.y);
        })
        // console.log('update scatterplot')
        // select('#point-canvas')
        //   .selectAll('path.scatter')
        //   .data(this.samples, d=>d.idx)
        //   .join(
        //     enter => enter.append('path')
        //         .classed('scatter', true)
        //         .attr('transform', d => `translate(${this.xScale(d.x)},${this.yScale(d.y)})`)
        //         .attr('d',  d => this.highlight_idx.has(d.idx) ? Shape.STAR : Shape.CIRCLE)
        //         .on('mouseover', (event, d) => { 
        //           this.add_highlight_idx({source: 'graph', idx: [d.idx]});
        //       }).on('mouseout', (event, d) => { 
        //         console.log('rm')
        //           this.rm_highlight_idx('graph');
        //       }),
        //     update => update
        //         .attr('transform', d => `translate(${this.xScale(d.x)},${this.yScale(d.y)})`)
        //         .attr('d',  d => this.highlight_idx.has(d.idx) ? Shape.STAR : Shape.CIRCLE)
        //   );
        //   this.init_lasso();
        // this.update_color();
      },
      update_color() {
        console.log('update colors', this.highlight.idx)
        select('#point-canvas')
        .selectAll('path.scatter')
        .attr('fill', this.ENABLE_OPACITY ? d => this.colors[d.idx] : d => this.scatter_colors[d.idx])
        .attr('opacity', this.ENABLE_OPACITY ? d => this.opacity[d.idx] : 1)
        .classed('hidden', d => !(d.visible = this.highlight.has(d.idx)))
        .classed('visible', d => d.visible)
        .filter(d => d.visible).raise();
        select('#point-canvas').select('.lasso').raise();
      },
      init_lasso() {
        if (this.brush !== null) return;
        if (this.all_samples.length === 0) return;
        console.log('init lasso')
        this.brush = 
          lasso()
          .closePathSelect(true)
          .closePathDistance(200)
          .items(selectAll("path.scatter"))
          .targetArea(select('#point-canvas'))
          .on("start", () => {
            selectAll("path.scatter").classed("not_possible", true);
          })
          .on("draw",  () => {
            this.brush.possibleItems().classed("not_possible", false)
            this.brush.notPossibleItems().classed("not_possible", true)
          })
          .on("end", () => {
            const selected_idxes = new Set(this.brush.selectedItems().data().map((x) => x.idx));
            if (selected_idxes.length > 0) {
              this.add_highlight_idx({highlight: this.highlight, source: 'lasso', idx: selected_idxes});
            } else {
              this.rm_highlight_idx({highlight: this.highlight, source: 'lasso'});
            }
          });
        select('#point-canvas').call(this.brush);
      },
    },
    computed: {
      ...mapState(['all_samples', 'highlight', 'scatter_colors', 'opacity', 'colors', 'line_attribute', 'meta_data']),
      samples() {
        return this.all_samples.filter(d => d);
      },
      highlight_idx() {
        console.log('graph.vue highlight_idx');
        return this.highlight.idx;
      },
      xRange() {
        console.log('graph.vue xRange');
        return extent(this.samples.filter(d => d), d => d.x);
      },
      yRange() {
        console.log('graph.vue yRange');
        return extent(this.samples.filter(d => d), d => d.y)
      },
      xScale() {
        console.log('graph.vue xScale');
        return scaleLinear()
          .rangeRound([0, this.viewWidth])
          .domain(this.xRange);
      },
      yScale() {
        console.log('graph.vue yScale');
        return scaleLinear()
          .rangeRound([0, this.viewWidth])
          .domain(this.yRange);
      },
      svgHeight() {
        return this.svgWidth;
      }
    },
    watch: {
      all_samples() {
        this.update_view();
      },
      highlight_idx() {
        this.update_view();
      },
      scatter_colors() {
        this.update_color();
      },
      line_attribute() {
        this.update_linechart();
      },
    }
  });
</script>

<style>
path.scatter.hidden { fill: #EEEEEE; }
.lasso path {
    stroke: rgb(80,80,80);
    stroke-width:1;
}

.lasso .drawn {
    fill-opacity:.05 ;
}

.lasso .loop_close {
    fill:none;
    stroke-dasharray: 1, 1;
}

.lasso .origin {
    fill:#3399FF;
    fill-opacity:.5;
}
</style>